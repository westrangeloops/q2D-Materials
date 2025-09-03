def get_atom_coordinates_by_index(data, atom_index):
    """
    Retrieves the x, y, z coordinates of an atom given its global index
    from the provided JSON data.

    Args:
        data (dict): The entire JSON data dictionary.
        atom_index (int): The global index of the atom to find.

    Returns:
        dict or None: A dictionary with 'x', 'y', 'z' coordinates if found,
                      otherwise None.
    """
    # Use the batch function for single index
    results = get_atom_coordinates_batch(data, [atom_index])
    return results.get(atom_index)


def get_atom_coordinates_batch(data, atom_indices):
    """
    Retrieves the x, y, z coordinates of multiple atoms given their global indices
    from the provided JSON data. Optimized for batch processing.

    Args:
        data (dict): The entire JSON data dictionary.
        atom_indices (list): List of global indices of atoms to find.

    Returns:
        dict: A dictionary mapping atom_index -> coordinates dict {'x', 'y', 'z'}
              Missing indices will not be present in the result.
    """
    # Convert to set for O(1) lookup and remove duplicates
    target_indices = set(atom_indices)
    results = {}
    
    # Single pass through octahedra to build results
    for octahedron_key, octahedron_info in data.get("octahedra", {}).items():
        # Check if we still have indices to find
        if not target_indices:
            break
            
        # Check the central atom
        central_atom = octahedron_info.get("central_atom")
        if central_atom:
            central_index = central_atom.get("global_index")
            if central_index in target_indices:
                coords = central_atom.get("coordinates")
                if coords:
                    results[central_index] = _format_coordinates(coords)
                    target_indices.remove(central_index)
        
        # Check ligand atoms
        ligand_indices = octahedron_info.get("ligand_atoms", {}).get("all_ligand_global_indices", [])
        ligand_coordinates = octahedron_info.get("detailed_atom_info", {}).get("ligand_coordinates", [])
        
        # Process ligands that match our target indices
        for i, ligand_index in enumerate(ligand_indices):
            if ligand_index in target_indices and i < len(ligand_coordinates):
                coords_list = ligand_coordinates[i]
                results[ligand_index] = _format_coordinates(coords_list)
                target_indices.remove(ligand_index)
                
                # Early exit if all indices found
                if not target_indices:
                    break
    
    return results


def _format_coordinates(coords):
    """
    Helper function to format coordinates into a consistent dict format.
    
    Args:
        coords: Can be a dict with x,y,z keys or a list/tuple with [x,y,z]
        
    Returns:
        dict: Dictionary with 'x', 'y', 'z' keys
    """
    if isinstance(coords, dict):
        return {"x": coords.get("x"), "y": coords.get("y"), "z": coords.get("z")}
    elif isinstance(coords, (list, tuple)) and len(coords) >= 3:
        return {"x": coords[0], "y": coords[1], "z": coords[2]}
    else:
        return {"x": None, "y": None, "z": None}


def create_atom_index_map(data):
    """
    Creates a precomputed index map for ultra-fast coordinate lookups.
    Useful when performing many coordinate lookups on the same dataset.

    Args:
        data (dict): The entire JSON data dictionary.

    Returns:
        dict: A dictionary mapping atom_index -> coordinates dict {'x', 'y', 'z'}
    """
    index_map = {}
    
    for octahedron_key, octahedron_info in data.get("octahedra", {}).items():
        # Map central atom
        central_atom = octahedron_info.get("central_atom")
        if central_atom:
            central_index = central_atom.get("global_index")
            coords = central_atom.get("coordinates")
            if central_index is not None and coords:
                index_map[central_index] = _format_coordinates(coords)
        
        # Map ligand atoms
        ligand_indices = octahedron_info.get("ligand_atoms", {}).get("all_ligand_global_indices", [])
        ligand_coordinates = octahedron_info.get("detailed_atom_info", {}).get("ligand_coordinates", [])
        
        for i, ligand_index in enumerate(ligand_indices):
            if i < len(ligand_coordinates):
                coords_list = ligand_coordinates[i]
                index_map[ligand_index] = _format_coordinates(coords_list)
    
    return index_map


def get_atom_coordinates_from_map(index_map, atom_indices):
    """
    Fast coordinate lookup using a precomputed index map.
    
    Args:
        index_map (dict): Precomputed map from create_atom_index_map()
        atom_indices (list or int): Single index or list of indices to look up
        
    Returns:
        dict or dict: If single index provided, returns coordinates dict or None.
                     If list provided, returns dict mapping indices to coordinates.
    """
    if isinstance(atom_indices, (int, float)):
        # Single index lookup
        return index_map.get(int(atom_indices))
    else:
        # Batch lookup
        return {idx: index_map[idx] for idx in atom_indices if idx in index_map}