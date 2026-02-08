"""
SMARTS Pattern Validator - Pattern matching using RDKit SMARTS.

This module provides SMARTS pattern matching and validation using RDKit,
converting results to NetworkX graph format for compatibility with the
graph-based architecture.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
import networkx as nx

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None

from ...utils.molecules.graph_converter import (
    graph_to_rdkit,
    rdkit_to_graph,
    validate_smiles
)

LOGGER = logging.getLogger(__name__)


def find_smarts_matches(
    graph: nx.Graph,
    smarts_pattern: str
) -> List[Dict[str, Any]]:
    """Find SMARTS pattern matches using RDKit.
    
    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph to search
    smarts_pattern : str
        SMARTS pattern string
        
    Returns
    -------
    List[Dict[str, Any]]
        List of match dictionaries, each containing:
        - 'mapping': Dict mapping RDKit indices to NetworkX indices
        - 'rdkit_match': Tuple of RDKit atom indices in the match
        
    Raises
    ------
    ImportError
        If RDKit is not available
    ValueError
        If SMARTS pattern is invalid
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SMARTS pattern matching")
    
    # Convert graph to RDKit
    mol = graph_to_rdkit(graph, preserve_coords=False)
    
    # Sanitize molecule (required before GetSubstructMatches)
    # This calculates implicit valence and prepares the molecule for pattern matching
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # If sanitization fails, try with relaxed options
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception:
            pass  # Continue anyway - some molecules may have unusual valences
    
    # Parse SMARTS pattern
    pattern = Chem.MolFromSmarts(smarts_pattern)
    if pattern is None:
        raise ValueError(f"Invalid SMARTS pattern: {smarts_pattern}")
    
    # Find matches
    matches = mol.GetSubstructMatches(pattern)
    
    # Convert back to NetworkX format
    result = []
    for match in matches:
        # Map RDKit indices back to NetworkX indices (fix: map rdkit_idx → nx_idx)
        nx_match = {
            rdkit_idx: graph.graph['rdkit_to_nx'][rdkit_idx]
            for rdkit_idx in match
        }
        result.append({'mapping': nx_match, 'rdkit_match': match})
    
    return result


def validate_smarts_pattern(
    graph: nx.Graph,
    smarts_pattern: str,
    validate_geometry: bool = True
) -> Tuple[bool, List[Dict], Optional[nx.Graph]]:
    """Validate SMARTS pattern on graph, return matches and validated graph.
    
    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph to validate
    smarts_pattern : str
        SMARTS pattern string
    validate_geometry : bool, default=True
        If True, return validated graph (not implemented - RDKit structures already validated)
        
    Returns
    -------
    Tuple[bool, List[Dict], Optional[nx.Graph]]
        (is_valid, matches, validated_graph)
        - is_valid: True if pattern matches found
        - matches: List of match dictionaries from find_smarts_matches()
        - validated_graph: Original graph (RDKit structures already validated)
    """
    matches = find_smarts_matches(graph, smarts_pattern)
    is_valid = len(matches) > 0
    
    # RDKit structures are already validated, so return original graph
    validated_graph = graph if validate_geometry else None
    
    return is_valid, matches, validated_graph

