"""
VASP Graph Background Processor

This module provides functionality to process VASP files and generate
interactive characterization graphs using the q2D_analyzer and
GraphHTMLExporter. It acts as a bridge between the API and the core
analysis functionality.

Functions
---------
process_vasp_file
    Main entry point: process VASP content and generate graph visualization
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import io

try:
    from ase.io import read
    from ase import Atoms
except ImportError:
    raise ImportError("ASE is required for VASP file processing")

try:
    from q2D_Materials.analyzer import q2D_analyzer
except ImportError:
    raise ImportError("q2D_Materials modules required for graph processing")


def process_vasp_file(
    vasp_content: str,
    analyze_params: Optional[Dict[str, Any]] = None,
    include_cavities: bool = True,
    cavity_metrics: bool = True,
    filename: str = "POSCAR",
) -> Dict[str, Any]:
    """
    Process VASP file content and generate Gephi GEXF visualization files.
    
    This function:
    1. Parses VASP content into an ASE Atoms object (supports .vasp, POSCAR, CONTCAR)
    2. Creates and analyzes the structure with q2D_analyzer
    3. Generates Gephi GEXF files for structure and cavities
    4. Returns graph data and paths to generated files
    
    Parameters
    ----------
    vasp_content : str
        The full text content of a VASP file
    analyze_params : dict, optional
        Parameters to pass to analyzer.analyze(). Examples:
        - cutoff_distance: float (default 4.0)
        - min_tolerance: float (default 0.2)
        - max_steps: int (default 20)
    include_cavities : bool, default=True
        Include cavity subgraphs in the export
    cavity_metrics : bool, default=True
        Compute and display cavity metrics
    filename : str, default="POSCAR"
        Original filename to determine format (POSCAR, CONTCAR, .vasp)
    
    Returns
    -------
    dict
        Dictionary containing:
        - status: 'success' or 'error'
        - message: Status message
        - graph_data: {nodes, edges, metadata} if successful
        - generated_files: List of paths to created GEXF files
        - cavities: List of cavity metadata if successful
        - error_details: Error message if failed
    
    Examples
    --------
    >>> vasp_str = open('structure.vasp').read()
    >>> result = process_vasp_file(vasp_str, filename='structure.vasp')
    >>> if result['status'] == 'success':
    ...     files = result['generated_files']
    """
    if analyze_params is None:
        analyze_params = {}
    
    try:
        # Step 1: Parse VASP content
        try:
            # Determine file format from filename
            if filename.endswith('.vasp'):
                suffix = '.vasp'
            elif filename == 'CONTCAR':
                suffix = '.CONTCAR'
            elif filename == 'POSCAR':
                suffix = '.POSCAR'
            else:
                suffix = '.vasp'  # Default fallback
            
            # Write to temporary file for ASE to read
            with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
                f.write(vasp_content)
                temp_path = f.name
            
            # Read structure - ASE will auto-detect format from file content
            structure = read(temp_path, format='vasp')
            Path(temp_path).unlink()  # Clean up temp file
            
        except Exception as e:
            return {
                'status': 'error',
                'message': 'Failed to parse VASP file',
                'error_details': str(e)
            }
        
        # Step 2: Create analyzer and load structure
        try:
            analyzer = q2D_analyzer(source=structure)
        except Exception as e:
            return {
                'status': 'error',
                'message': 'Failed to create analyzer',
                'error_details': str(e)
            }
        
        # Step 3: Analyze the structure
        try:
            analyzer.analyze(**analyze_params)
        except Exception as e:
            return {
                'status': 'error',
                'message': 'Structure analysis failed',
                'error_details': str(e)
            }
        
        # Step 4: Generate Gephi visualization files
        try:
            # Create temporary directory for GEXF output
            temp_dir = tempfile.mkdtemp()
            output_path = str(Path(temp_dir) / f"{filename}_structure.gexf")
            
            # Export graph to Gephi GEXF
            generated_files = analyzer.export_graph_gephi(
                output_path=output_path,
                include_cavities=include_cavities,
                cavity_metrics=cavity_metrics
            )
            
        except Exception as e:
            return {
                'status': 'error',
                'message': 'Failed to generate Gephi files',
                'error_details': str(e)
            }
        
        # Step 5: Extract graph data from analyzer
        try:
            graph = analyzer.get_graph()
            
            # Build nodes list (exclude cavity nodes by default for main graph)
            nodes = []
            node_id_map = {}
            for i, (node_id, node_data) in enumerate(graph.nodes(data=True)):
                # Skip cavity nodes for main graph view
                if node_data.get('node_type') == 'cavity':
                    continue
                node_id_map[node_id] = i
                node_type = node_data.get('node_type', 'unknown')
                nodes.append({
                    'id': i,
                    'label': str(node_id).replace('_', ' '),
                    'node_type': node_type,
                    'original_id': node_id,
                })
            
            # Build edges list (exclude cavity edges for main graph)
            edges = []
            for source, target, edge_data in graph.edges(data=True):
                # Skip edges connected to cavities for main graph view
                source_data = graph.nodes[source]
                target_data = graph.nodes[target]
                if source_data.get('node_type') == 'cavity' or target_data.get('node_type') == 'cavity':
                    continue
                if source in node_id_map and target in node_id_map:
                    edge_type = edge_data.get('edge_type', 'unknown')
                    edges.append({
                        'from': node_id_map[source],
                        'to': node_id_map[target],
                        'edge_type': edge_type,
                    })
            
            # Build metadata
            metadata = {
                'formula': structure.get_chemical_formula(),
                'atom_count': len(structure),
                'structure_type': analyzer.structure_type,
                'octahedra_count': len(analyzer.get_octahedra()),
                'layer_count': len(analyzer.get_layers()),
                'node_count': len(nodes),
                'edge_count': len(edges),
                'spacers_count': len(analyzer.get_spacers()),
                'a_sites_count': len(analyzer.get_a_sites()),
            }
            
            # Extract cavity metadata (basic info without subgraphs)
            cavities_meta = []
            try:
                cavities = analyzer.get_cavities()
                for idx, cavity in enumerate(cavities):
                    cavities_meta.append({
                        'id': cavity.get('id', f'cavity_{idx}'),
                        'index': idx,
                        'contains_a_site': cavity.get('contains_a_site', False),
                        'octahedra_count': len(cavity.get('octahedra_indices', [])),
                        'x_atom_count': len(cavity.get('x_atom_indices', [])),
                        'center_position': cavity.get('center_position'),
                        'a_site_type': cavity.get('a_site_type'),
                    })
            except Exception as e:
                import sys
                print(f"Warning: Could not extract cavity metadata: {e}", file=sys.stderr)
                cavities_meta = []
            
        except Exception as e:
            return {
                'status': 'error',
                'message': 'Failed to extract graph data',
                'error_details': str(e)
            }
        
        # Return success response
        return {
            'status': 'success',
            'message': f'Successfully analyzed structure with {len(structure)} atoms',
            'graph_data': {
                'nodes': nodes,
                'edges': edges,
                'metadata': metadata,
            },
            'cavities': cavities_meta,
            'generated_files': generated_files,
        }
        
    except Exception as e:
        # Catch-all for unexpected errors
        return {
            'status': 'error',
            'message': 'Unexpected error during processing',
            'error_details': str(e)
        }


def get_cavity_subgraph(
    analyzer,
    cavity_index: int,
) -> Dict[str, Any]:
    """
    Get subgraph data for a specific cavity.
    
    This function retrieves the subgraph for a single cavity, converting it to
    vis.js format for visualization.
    
    Parameters
    ----------
    analyzer : q2D_analyzer
        The analyzer object with analyzed structure
    cavity_index : int
        Index of the cavity to retrieve (0-based)
    
    Returns
    -------
    dict
        Dictionary containing:
        - status: 'success' or 'error'
        - message: Status message
        - nodes: List of nodes in cavity subgraph
        - edges: List of edges in cavity subgraph
        - cavity_data: Cavity metadata
        - error_details: Error message if failed
    
    Examples
    --------
    >>> analyzer = q2D_analyzer('structure.vasp')
    >>> analyzer.analyze()
    >>> result = get_cavity_subgraph(analyzer, 0)
    >>> if result['status'] == 'success':
    ...     nodes = result['nodes']
    ...     edges = result['edges']
    """
    try:
        # Get all cavities
        cavities = analyzer.get_cavities()
        
        if cavity_index >= len(cavities):
            return {
                'status': 'error',
                'message': f'Cavity index {cavity_index} out of range (total: {len(cavities)})',
            }
        
        cavity_data = cavities[cavity_index]
        cavity_subgraph = cavity_data.get('subgraph')
        
        if cavity_subgraph is None:
            return {
                'status': 'error',
                'message': f'No subgraph available for cavity {cavity_index}',
            }
        
        # Build nodes list for cavity subgraph
        nodes = []
        node_id_map = {}
        for i, (node_id, node_data) in enumerate(cavity_subgraph.nodes(data=True)):
            node_id_map[node_id] = i
            node_type = node_data.get('node_type', 'unknown')
            nodes.append({
                'id': i,
                'label': str(node_id).replace('_', ' '),
                'node_type': node_type,
                'original_id': node_id,
            })
        
        # Build edges list for cavity subgraph
        edges = []
        for source, target, edge_data in cavity_subgraph.edges(data=True):
            if source in node_id_map and target in node_id_map:
                edge_type = edge_data.get('edge_type', 'unknown')
                edges.append({
                    'from': node_id_map[source],
                    'to': node_id_map[target],
                    'edge_type': edge_type,
                })
        
        return {
            'status': 'success',
            'message': f'Retrieved cavity {cavity_index} subgraph',
            'nodes': nodes,
            'edges': edges,
            'cavity_data': {
                'id': cavity_data.get('id'),
                'index': cavity_index,
                'contains_a_site': cavity_data.get('contains_a_site', False),
                'octahedra_count': len(cavity_data.get('octahedra_indices', [])),
                'x_atom_count': len(cavity_data.get('x_atom_indices', [])),
                'center_position': cavity_data.get('center_position'),
                'a_site_type': cavity_data.get('a_site_type'),
            }
        }
    
    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'message': 'Failed to get cavity subgraph',
            'error_details': str(e),
            'traceback': traceback.format_exc()
        }


# Note: VASP format validation is handled by ASE's read() function
# which will raise appropriate errors for invalid VASP files
