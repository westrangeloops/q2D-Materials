"""Molecule candidate analysis for DJ and RP spacer suitability.

This module provides functions for analyzing whether molecules are suitable as
DJ (Dion-Jacobson) or RP (Ruddlesden-Popper) spacers based on pattern matching
with user-defined SMILES patterns.
"""

from typing import Union, Optional, List, Tuple, Set, Dict, Any
from dataclasses import dataclass
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from ase import Atoms
from collections import Counter

from ..utils.pymatgen_utils import build_molecular_graph
from ..utils.graph_utils import find_shortest_path, validate_path_continuity, get_connected_components
from ...utils.molecules.graph_converter import (
    graph_to_rdkit,
    rdkit_to_graph,
    validate_smiles,
    atoms_to_graph
)
from ...builders.optimizers import elongate_molecule
from ...modifier.fragment import from_smiles, validate_fragment
from .smarts_validator import find_smarts_matches

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None

# Default allowed elements in backbone path (between terminal groups)
# Users can customize via API parameters
DEFAULT_ALLOWED_BACKBONE_ELEMENTS: Set[str] = {'C', 'N', 'O'}
# Default forbidden elements in backbone path
# Users can customize via API parameters
DEFAULT_FORBIDDEN_BACKBONE_ELEMENTS: Set[str] = {'P', 'Si', 'B', 'Se', 'I', 'As', 'Ge', 'Sn', 'Pb', 'Bi', 'Al', 'Ti', 'Fe', 'Cu', 'Zn'}
# Default maximum allowed ratio of non-carbon heavy atoms in backbone (excluding H, N)
# Users can customize via API parameters
DEFAULT_MAX_NON_CARBON_RATIO: float = 0.3

# Default SMILES patterns for terminal groups
# Users can customize via API parameters
# Note: RDKit requires explicit notation, so [NH2]C instead of NH2C
DEFAULT_INITIAL_PATTERN: str = '[NH2]C'  # NH2 bonded to carbon
DEFAULT_FINAL_PATTERN: str = '[NH2]C'   # NH2 bonded to carbon

@dataclass
class TerminalGroup:
    """Represents a terminal NH2 or NH3 group.

    Attributes
    ----------
    n_index : int
        Nitrogen atom index in the molecule
    group_type : str
        Type of group: "NH2" or "NH3"
    h_indices : List[int]
        Indices of hydrogen atoms bonded to nitrogen
    carbon_neighbor : Optional[int]
        Index of carbon atom bonded to nitrogen (None if not found)
    """
    n_index: int
    group_type: str
    h_indices: List[int]
    carbon_neighbor: Optional[int]


@dataclass
class SpacerCandidateResult:
    """Results from spacer candidate analysis (DJ or RP).

    Attributes
    ----------
    spacer_type : str
        Type of spacer: "DJ" or "RP"
    is_valid : bool
        Whether molecule is suitable as the specified spacer type
    reason : str
        Explanation of validity or failure reason
    terminal_groups : List[TerminalGroup]
        Detected NH2/NH3 terminal groups
    valid_paths : List[List[int]]
        Valid paths from NH2/3-C to C-NH2/3 (only for DJ spacers)
    original_atoms : Atoms
        Input molecule as ASE Atoms object
    graph : nx.Graph
        Molecular connectivity graph
    """
    spacer_type: str
    is_valid: bool
    reason: str
    terminal_groups: List[TerminalGroup]
    valid_paths: List[List[int]]
    original_atoms: Atoms
    graph: nx.Graph


# ============================================================================
# PATTERN MATCHING FUNCTIONS (Pattern-Based Validation)
# ============================================================================

def _parse_pattern(pattern_smarts: str) -> Any:
    """
    Parse SMARTS pattern into RDKit molecule.

    Parameters
    ----------
    pattern_smarts : str
        SMARTS string defining the terminal pattern

    Returns
    -------
    Chem.Mol
        RDKit molecule object for the pattern
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for pattern parsing")
    
    try:
        pattern_mol = Chem.MolFromSmarts(pattern_smarts)
        if pattern_mol is None:
            raise ValueError(f"RDKit could not parse SMARTS pattern: {pattern_smarts}")
        return pattern_mol
    except Exception as e:
        raise ValueError(f"Invalid SMARTS pattern '{pattern_smarts}': {e}")


def _find_pattern_matches(
    molecule_graph: nx.Graph,
    pattern_mol: Any
) -> List[Dict]:
    """
    Find all subgraph matches using RDKit SMARTS matching.

    Workflow:
    1. Graph -> RDKit Mol (ASE -> Graph done by caller)
    2. Patch charges/topology on RDKit Mol (Geometric/Topological analysis)
    3. SMARTS matching on RDKit Mol
    4. Map matches back to Graph indices

    Identifies "anchor" atoms dynamically: an anchor is an atom in the matched
    pattern that connects to the rest of the molecule (the backbone).
    
    **Terminal Group Filtering**: Only matches with exactly one anchor atom are
    returned. This ensures the pattern represents a terminal group attached to
    a backbone, rather than a substructure within the backbone.

    Parameters
    ----------
    molecule_graph : nx.Graph
        Full molecular connectivity graph (graph-based, no 3D coordinates needed)
    pattern_mol : Chem.Mol
        RDKit molecule object for the SMARTS pattern

    Returns
    -------
    List[Dict]
        List of matches: [{'mapping': {pattern_idx: mol_idx}, 'anchor_idx': mol_anchor_idx}, ...]
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for pattern matching")
    
    # Convert molecule graph to RDKit Mol
    mol = graph_to_rdkit(molecule_graph, preserve_coords=False)
    
    # PATCH: Fix N charges if missing (common issue when converting from Atoms without charge inference)
    # Ammonium nitrogens (N bonded to 4 atoms) should have +1 charge
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 0:
                # If N has 4 neighbors (e.g. 1 C and 3 H), it should be N+
                if atom.GetDegree() == 4:
                    atom.SetFormalCharge(1)

    # Sanitize molecules before pattern matching (required for RDKit)
    try:
        Chem.SanitizeMol(mol)
    except:
        pass  # Some molecules may fail sanitization, but we can still try matching
    
    # Find matches using RDKit
    # GetSubstructMatches returns tuples of (mol_atom_idx, ...) matching pattern atoms
    matches_rdkit = mol.GetSubstructMatches(pattern_mol)
    
    # Get or create mapping dictionaries from graphs
    # If graph was built from Atoms (not SMILES), the mapping won't exist, so create it
    mol_rdkit_to_nx = molecule_graph.graph.get('rdkit_to_nx', {})
    if not mol_rdkit_to_nx:
        # Create mapping by sorting nodes the same way graph_to_rdkit does
        sorted_mol_nodes = sorted(molecule_graph.nodes())
        mol_rdkit_to_nx = {rdkit_idx: nx_idx for rdkit_idx, nx_idx in enumerate(sorted_mol_nodes)}
        # Store it in the graph for future use
        molecule_graph.graph['rdkit_to_nx'] = mol_rdkit_to_nx

    matches = []
    for rdkit_match in matches_rdkit:
        # Convert match to set of NX indices
        match_nx_indices = set()
        mapping = {}
        
        for pat_idx, mol_rdkit_idx in enumerate(rdkit_match):
            mol_nx_idx_mapped = mol_rdkit_to_nx.get(mol_rdkit_idx)
            if mol_nx_idx_mapped is not None:
                match_nx_indices.add(mol_nx_idx_mapped)
                mapping[pat_idx] = mol_nx_idx_mapped
        
        # Identify anchors: atoms in the match that have neighbors NOT in the match
        anchors = []
        for mol_nx_idx in match_nx_indices:
            for neighbor in molecule_graph.neighbors(mol_nx_idx):
                if neighbor not in match_nx_indices:
                    anchors.append(mol_nx_idx)
                    break  # Found one external neighbor, this atom is an anchor
        
        # Filter for terminal groups: must have exactly one anchor connecting to backbone
        if len(anchors) == 1:
            matches.append({
                'mapping': mapping,
                'anchor_idx': anchors[0]
            })
        elif len(anchors) == 0:
            # Match covers the entire molecule (e.g. MA matching [NH3+]C)
            # Try to find a Carbon atom to serve as the anchor
            anchor = None
            for mol_idx in match_nx_indices:
                if molecule_graph.nodes[mol_idx].get('symbol') == 'C':
                    anchor = mol_idx
                    break
            
            # If no Carbon, use the first atom
            if anchor is None and match_nx_indices:
                anchor = sorted(list(match_nx_indices))[0]
                
            if anchor is not None:
                matches.append({
                    'mapping': mapping,
                    'anchor_idx': anchor
                })
        else:
            # Multiple anchors: pattern matches entire molecule or most of it
            # (e.g. C[NH3+] where both C and N have external neighbors)
            # Select Carbon as anchor if present, otherwise use first anchor
            anchor = None
            for mol_idx in anchors:
                if molecule_graph.nodes[mol_idx].get('symbol') == 'C':
                    anchor = mol_idx
                    break
            
            # If no Carbon in anchors, try to find Carbon in match
            if anchor is None:
                for mol_idx in match_nx_indices:
                    if molecule_graph.nodes[mol_idx].get('symbol') == 'C':
                        anchor = mol_idx
                        break
            
            # If still no Carbon, use first anchor
            if anchor is None and anchors:
                anchor = anchors[0]
                
            if anchor is not None:
                matches.append({
                    'mapping': mapping,
                    'anchor_idx': anchor
                })
    
    return matches


def _find_valid_paths_pattern_based(
    graph: nx.Graph,
    initial_matches: List[Dict],
    final_matches: List[Dict],
    min_length: int,
    allowed_backbone_elements: Optional[Set[str]] = None,
    forbidden_backbone_elements: Optional[Set[str]] = None,
    max_non_carbon_ratio: Optional[float] = None,
    allow_same_carbon: bool = False
) -> List[List[int]]:
    """
    Find valid paths between pattern matches.

    Similar to _find_valid_paths but uses pattern matches instead of
    heuristic terminal groups. Validates path composition and connectivity.

    Parameters
    ----------
    graph : nx.Graph
        Molecular connectivity graph
    initial_matches : List[Dict]
        Pattern matches for initial terminal
    final_matches : List[Dict]
        Pattern matches for final terminal
    min_length : int
        Minimum number of atoms in path between carbons
    atoms : Atoms, optional
        ASE Atoms object (deprecated - symbols now come from graph nodes)
        Kept for backward compatibility but not used
    allowed_backbone_elements : Set[str], optional
        Set of allowed elements in backbone path
    forbidden_backbone_elements : Set[str], optional
        Set of forbidden elements in backbone path
    max_non_carbon_ratio : float, optional
        Maximum ratio of non-carbon heavy atoms in backbone
    allow_same_carbon : bool
        Whether to allow both patterns to match the same carbon

    Returns
    -------
    List[List[int]]
        Valid paths as lists of atom indices
    """
    # Use defaults if not specified
    if allowed_backbone_elements is None:
        allowed_backbone_elements = DEFAULT_ALLOWED_BACKBONE_ELEMENTS
    if forbidden_backbone_elements is None:
        forbidden_backbone_elements = DEFAULT_FORBIDDEN_BACKBONE_ELEMENTS
    if max_non_carbon_ratio is None:
        max_non_carbon_ratio = DEFAULT_MAX_NON_CARBON_RATIO

    if not nx.is_connected(graph):
        # Graph has disconnected fragments - no valid paths possible
        return []

    valid_paths = []
    # Get symbols from graph nodes (graph-based matching - no Atoms object needed)
    # Create a mapping from node index to symbol
    node_to_symbol = {node: graph.nodes[node].get('symbol', 'C') for node in graph.nodes()}
    
    # atoms parameter is deprecated - symbols come from graph nodes

    for init_match in initial_matches:
        for final_match in final_matches:
            c1 = init_match['anchor_idx']
            c2 = final_match['anchor_idx']

            # Skip if either pattern has no carbon
            if c1 is None or c2 is None:
                continue

            # Check carbon uniqueness constraint
            if not allow_same_carbon and c1 == c2:
                continue

            # Find shortest path from c1 to c2 using shared graph utilities
            path = find_shortest_path(graph, c1, c2)
            if path is None:
                continue

            # Validate path length
            if len(path) < min_length:
                continue

            # Verify path continuity using shared graph utilities
            if not validate_path_continuity(graph, path):
                continue

            # Validate backbone composition (get symbols from graph nodes)
            path_symbols = [node_to_symbol.get(idx, 'C') for idx in path]

            # Check for forbidden elements in backbone
            forbidden_in_path = set(path_symbols) & forbidden_backbone_elements
            if forbidden_in_path:
                continue

            # Check for disallowed elements in backbone
            disallowed_in_path = set(path_symbols) - allowed_backbone_elements - {'H'}
            if disallowed_in_path:
                continue

            # Check carbon ratio (exclude H and N from heavy atom count)
            heavy_atoms = [s for s in path_symbols if s not in ['H', 'N']]
            if heavy_atoms:
                carbon_count = sum(1 for s in heavy_atoms if s == 'C')
                non_carbon_ratio = 1.0 - (carbon_count / len(heavy_atoms))
                if non_carbon_ratio > max_non_carbon_ratio:
                    continue

            # Path is valid
            valid_paths.append(path)

    return valid_paths


def analyze_molecule_candidate(
    molecule: Union[Atoms, str],
    spacer_type: str = "DJ",
    initial_pattern: Union[str, List[str]] = None,
    final_pattern: Union[str, List[str]] = None,
    min_chain_length: int = 2,
    allowed_backbone_elements: Optional[Set[str]] = None,
    forbidden_backbone_elements: Optional[Set[str]] = None,
    max_non_carbon_ratio: Optional[float] = None,
    allow_same_carbon: bool = False
) -> SpacerCandidateResult:
    """Analyze if molecule is suitable as DJ or RP spacer using pattern matching.

    **Pattern-Based Validation**: Users specify SMILES patterns that define
    valid terminal groups. The system finds all matches and validates paths
    between them.

    - DJ (Dion-Jacobson): Bifunctional spacer with paths between two patterns
    - RP (Ruddlesden-Popper): Monofunctional spacer with at least one pattern match

    Parameters
    ----------
    molecule : Atoms or str
        ASE Atoms object or SMILES string
    spacer_type : str
        Type of spacer to analyze: "DJ" or "RP" (default: "DJ")
    initial_pattern : str or List[str], optional
        SMILES pattern(s) for initial terminal (default: 'NH2C')
        Examples: '[NH3+]C', 'NH2C', ['[NH3+]C', 'NH2C']
    final_pattern : str or List[str], optional
        SMILES pattern(s) for final terminal (default: 'NH2C')
        Can be different from initial_pattern for asymmetric spacers
    min_chain_length : int
        Minimum atoms between terminal carbons for DJ (default: 2)
        Ignored for RP spacers
    allowed_backbone_elements : Set[str], optional
        Elements allowed in backbone path (default: {'C', 'N', 'O'})
    forbidden_backbone_elements : Set[str], optional
        Elements forbidden in backbone path (default: {'P', 'Si', 'B', metals...})
    max_non_carbon_ratio : float, optional
        Maximum ratio of non-carbon heavy atoms in backbone (default: 0.3)
    allow_same_carbon : bool
        Whether terminals can share the same carbon (default: False)

    Returns
    -------
    SpacerCandidateResult
        Analysis with validity, terminal groups, and paths (DJ only)

    Examples
    --------
    >>> from q2D_Materials.analyzer import q2D_analyzer
    >>> analyzer = q2D_analyzer()
    >>>
    >>> # Basic usage with NH2 groups
    >>> result = analyzer.mol_validate(
    ...     "NCCCCN",
    ...     spacer_type="DJ",
    ...     initial_pattern='NH2C',
    ...     final_pattern='NH2C'
    ... )
    >>>
    >>> # Ammonium terminals
    >>> result = analyzer.mol_validate(
    ...     molecule,
    ...     spacer_type="DJ",
    ...     initial_pattern='[NH3+]C',
    ...     final_pattern='[NH3+]C'
    ... )
    >>>
    >>> # Multiple final patterns
    >>> result = analyzer.mol_validate(
    ...     molecule,
    ...     spacer_type="DJ",
    ...     initial_pattern='[NH3+]C',
    ...     final_pattern=['[NH3+]C', 'NH2C']
    ... )
    """
    spacer_type = spacer_type.upper()

    # Use default patterns if not specified
    if initial_pattern is None:
        initial_pattern = DEFAULT_INITIAL_PATTERN
    if final_pattern is None:
        final_pattern = DEFAULT_FINAL_PATTERN

    # Convert patterns to lists for uniform handling
    if isinstance(initial_pattern, str):
        initial_patterns = [initial_pattern]
    else:
        initial_patterns = initial_pattern

    if isinstance(final_pattern, str):
        final_patterns = [final_pattern]
    else:
        final_patterns = final_pattern

    # Convert SMILES to graph directly (no 3D coordinates needed for pattern matching)
    if isinstance(molecule, str):
        try:
            # Use RDKit for SMILES parsing - fast, graph-based, no 3D coordinates
            validate_smiles(molecule)
            mol = Chem.MolFromSmiles(molecule)
            if mol is None:
                raise ValueError(f"RDKit could not parse SMILES: {molecule}")
            graph = rdkit_to_graph(mol, coords=None)
            # Create empty Atoms object for result (not needed for graph-based matching)
            atoms = Atoms()
        except Exception as e:
            return SpacerCandidateResult(
                spacer_type=spacer_type,
                is_valid=False,
                reason=f"Invalid SMILES string: {e}",
                terminal_groups=[],
                valid_paths=[],
                original_atoms=Atoms(),
                graph=nx.Graph()
            )
    else:
        # If Atoms object provided, convert to graph using unified converter
        # This ensures both SMILES and Atoms inputs produce graphs with sequential node indices
        atoms = molecule
        try:
            graph = atoms_to_graph(atoms, preserve_coords=True)
        except Exception as e:
            return SpacerCandidateResult(
                spacer_type=spacer_type,
                is_valid=False,
                reason=f"Failed to build molecular graph: {e}",
                terminal_groups=[],
                valid_paths=[],
                original_atoms=atoms,
                graph=nx.Graph()
            )

    # Parse patterns and find matches
    try:
        # Find all initial pattern matches
        all_initial_matches = []
        for pat in initial_patterns:
            pat_mol = _parse_pattern(pat)
            matches = _find_pattern_matches(graph, pat_mol)
            all_initial_matches.extend(matches)

        # Find all final pattern matches
        all_final_matches = []
        for pat in final_patterns:
            pat_mol = _parse_pattern(pat)
            matches = _find_pattern_matches(graph, pat_mol)
            all_final_matches.extend(matches)

    except ValueError as e:
        return SpacerCandidateResult(
            spacer_type=spacer_type,
            is_valid=False,
            reason=f"Pattern parsing error: {e}",
            terminal_groups=[],
            valid_paths=[],
            original_atoms=atoms,
            graph=graph
        )

    # Check if patterns were found
    if len(all_initial_matches) == 0:
        return SpacerCandidateResult(
            spacer_type=spacer_type,
            is_valid=False,
            reason=f"No NH terminal groups found (no matches for pattern(s): {initial_patterns})",
            terminal_groups=[],
            valid_paths=[],
            original_atoms=atoms,
            graph=graph
        )

    # Type-specific validation
    if spacer_type == "DJ":
        # DJ spacers need final pattern matches
        if len(all_final_matches) == 0:
            return SpacerCandidateResult(
                spacer_type=spacer_type,
                is_valid=False,
                reason=f"No matches found for final pattern(s): {final_patterns}",
                terminal_groups=[],
                valid_paths=[],
                original_atoms=atoms,
                graph=graph
            )

        # Find valid paths between pattern matches
        valid_paths = _find_valid_paths_pattern_based(
            graph,
            all_initial_matches,
            all_final_matches,
            min_chain_length,
            allowed_backbone_elements,
            forbidden_backbone_elements,
            max_non_carbon_ratio,
            allow_same_carbon
        )

        if len(valid_paths) == 0:
            # Check if we have distinct terminals (for better error message)
            initial_carbons = {m['anchor_idx'] for m in all_initial_matches if m['anchor_idx'] is not None}
            final_carbons = {m['anchor_idx'] for m in all_final_matches if m['anchor_idx'] is not None}
            distinct_terminals = len(initial_carbons | final_carbons)
            
            if distinct_terminals < 2:
                reason = f"DJ spacer requires at least 2 distinct terminal groups, found {distinct_terminals}"
            else:
                reason = "No valid backbone path found between pattern matches"
            
            return SpacerCandidateResult(
                spacer_type=spacer_type,
                is_valid=False,
                reason=reason,
                terminal_groups=[],
                valid_paths=[],
                original_atoms=atoms,
                graph=graph
            )

        # Success!
        return SpacerCandidateResult(
            spacer_type=spacer_type,
            is_valid=True,
            reason=f"Valid DJ spacer with {len(valid_paths)} path(s) between patterns",
            terminal_groups=[],
            valid_paths=valid_paths,
            original_atoms=atoms,
            graph=graph
        )

    elif spacer_type == "RP":
        # RP spacers need at least one match with carbon
        matches_with_carbon = [m for m in all_initial_matches if m['anchor_idx'] is not None]
        if len(matches_with_carbon) == 0:
            return SpacerCandidateResult(
                spacer_type=spacer_type,
                is_valid=False,
                reason="No pattern matches with carbon attachment found",
                terminal_groups=[],
                valid_paths=[],
                original_atoms=atoms,
                graph=graph
            )

        # Success!
        return SpacerCandidateResult(
            spacer_type=spacer_type,
            is_valid=True,
            reason=f"Valid RP spacer with {len(all_initial_matches)} pattern match(es)",
            terminal_groups=[],
            valid_paths=[],
            original_atoms=atoms,
            graph=graph
        )

    else:
        return SpacerCandidateResult(
            spacer_type=spacer_type,
            is_valid=False,
            reason=f"Unknown spacer_type '{spacer_type}'. Use 'DJ' or 'RP'.",
            terminal_groups=[],
            valid_paths=[],
            original_atoms=atoms,
            graph=graph
        )


def analyze_dj_candidate(
    molecule: Union[Atoms, str],
    initial_pattern: Union[str, List[str]] = None,
    final_pattern: Union[str, List[str]] = None,
    min_chain_length: int = 2,
    allowed_backbone_elements: Optional[Set[str]] = None,
    forbidden_backbone_elements: Optional[Set[str]] = None,
    max_non_carbon_ratio: Optional[float] = None,
    allow_same_carbon: bool = False
) -> SpacerCandidateResult:
    """Analyze if molecule is suitable as a DJ spacer.

    This is a wrapper for analyze_molecule_candidate with spacer_type="DJ".
    See analyze_molecule_candidate for parameter documentation.
    """
    return analyze_molecule_candidate(
        molecule,
        spacer_type="DJ",
        initial_pattern=initial_pattern,
        final_pattern=final_pattern,
        min_chain_length=min_chain_length,
        allowed_backbone_elements=allowed_backbone_elements,
        forbidden_backbone_elements=forbidden_backbone_elements,
        max_non_carbon_ratio=max_non_carbon_ratio,
        allow_same_carbon=allow_same_carbon
    )


def analyze_rp_candidate(
    molecule: Union[Atoms, str],
    initial_pattern: Union[str, List[str]] = None
) -> SpacerCandidateResult:
    """Analyze if molecule is suitable as an RP spacer.

    This is a wrapper for analyze_molecule_candidate with spacer_type="RP".
    See analyze_molecule_candidate for parameter documentation.
    """
    return analyze_molecule_candidate(
        molecule,
        spacer_type="RP",
        initial_pattern=initial_pattern
    )


def convert_nh2_to_nh3(atoms: Atoms, n_index: int) -> Atoms:
    """Convert NH2 group to NH3 by adding hydrogen atom.

    Adds H atom with proper tetrahedral geometry (sp3, 109.5° angles).

    Parameters
    ----------
    atoms : Atoms
        Molecule with NH2 group
    n_index : int
        Nitrogen atom index to convert

    Returns
    -------
    Atoms
        Modified molecule with NH3 group

    Raises
    ------
    ValueError
        If nitrogen doesn't have exactly 2 H neighbors
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    # Build graph to find H neighbors using unified converter
    graph = atoms_to_graph(atoms, preserve_coords=True)

    # Find H atoms bonded to N
    n_pos = positions[n_index]
    h_indices = []

    for neighbor in graph.neighbors(n_index):
        if symbols[neighbor] == 'H':
            h_pos = positions[neighbor]
            dist = np.linalg.norm(n_pos - h_pos)
            if 0.9 < dist < 1.2:
                h_indices.append(neighbor)

    if len(h_indices) != 2:
        raise ValueError(f"Expected NH2 (2 H atoms), found {len(h_indices)} at N index {n_index}")

    # Get positions of existing H atoms
    h1_pos = positions[h_indices[0]]
    h2_pos = positions[h_indices[1]]

    # Calculate vectors from N to H
    v1 = h1_pos - n_pos
    v2 = h2_pos - n_pos

    # Calculate normal to H-N-H plane
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    # Ideal N-H bond length
    nh_bond_length = 1.01

    # Calculate position for third H using tetrahedral geometry
    # The third H should be on the opposite side of the H-N-H plane
    bisector = (v1 + v2) / 2
    # Position new H to maintain tetrahedral angle
    v3 = -bisector + normal * (nh_bond_length * 0.8)
    v3 = v3 / np.linalg.norm(v3) * nh_bond_length

    new_h_pos = n_pos + v3

    # Create new Atoms object with added H
    new_atoms = atoms.copy()
    new_atoms.append(Atoms('H', positions=[new_h_pos]))

    return new_atoms


def clean_molecule(
    atoms: Atoms,
    patterns: Union[str, List[str]] = None,
    convert_nh2_to_nh3_flag: bool = True,
    remove_fragments: bool = True
) -> Atoms:
    """Clean molecule by converting NH2→NH3 and removing salt/fragment impurities.

    This function:
    1. Identifies the main molecular fragment containing pattern matches
    2. Removes disconnected fragments (salts like Cl, Br, counter-ions, etc.)
    3. Optionally converts all NH2 groups to NH3 for consistency

    Parameters
    ----------
    atoms : Atoms
        Molecule to clean
    patterns : str or List[str], optional
        SMILES patterns to search for (default: ['NH2C', '[NH3+]C'])
        Used to identify the main fragment
    convert_nh2_to_nh3_flag : bool
        If True, converts all NH2 groups to NH3 (default: True)
    remove_fragments : bool
        If True, removes disconnected fragments/salts (default: True)

    Returns
    -------
    Atoms
        Cleaned molecule with only the main fragment and NH3 groups

    Raises
    ------
    ValueError
        If no pattern matches found in any fragment

    Examples
    --------
    >>> # Molecule with Br salt: "NCCCCBr" → cleaned to "NCCCC" with NH3
    >>> cleaned = clean_molecule(molecule_with_salt)
    >>> # Multi-fragment: "NCCCCN.Cl.Br" → "NCCCCN" with both NH3
    >>> cleaned = clean_molecule(multi_fragment)
    """
    # Use default patterns if not specified
    if patterns is None:
        patterns = ['NH2C', '[NH3+]C']
    elif isinstance(patterns, str):
        patterns = [patterns]

    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    # Build molecular graph using unified converter
    graph = atoms_to_graph(atoms, preserve_coords=True)

    # Find all connected components using shared graph utilities
    components = get_connected_components(graph)

    if remove_fragments and len(components) > 1:
        # Find the component with most pattern matches
        main_component = None
        max_matches = 0

        for component in components:
            # Create subgraph for this component
            subgraph = graph.subgraph(component)

            # Count pattern matches in this component
            component_matches = 0
            for pattern in patterns:
                try:
                    pat_mol = _parse_pattern(pattern)
                    matches = _find_pattern_matches(subgraph, pat_mol)
                    component_matches += len(matches)
                except:
                    continue

            if component_matches > max_matches:
                max_matches = component_matches
                main_component = sorted(list(component))

        if main_component is None:
            raise ValueError(f"No pattern matches found for patterns: {patterns}")

        # Extract only the main component atoms
        main_symbols = [symbols[i] for i in main_component]
        main_positions = [positions[i] for i in main_component]
        cleaned_atoms = Atoms(main_symbols, positions=main_positions)

    else:
        # Keep all atoms
        cleaned_atoms = atoms.copy()

    # Convert NH2 → NH3 if requested
    if convert_nh2_to_nh3_flag:
        # Find all NH2 pattern matches
        try:
            # Rebuild graph for cleaned atoms using unified converter
            graph = atoms_to_graph(cleaned_atoms, preserve_coords=True)

            # Search for NH2 groups using fast direct parsing
            pat_mol = _parse_pattern('[NH2]C')
            matches = _find_pattern_matches(graph, pat_mol)

            # Find nitrogen indices from matches
            nh2_nitrogen_indices = set()
            for match in matches:
                # Find nitrogen node in the match mapping
                for pat_idx, mol_node in match['mapping'].items():
                    if pat_mol.GetAtomWithIdx(pat_idx).GetSymbol() == 'N':
                        nh2_nitrogen_indices.add(mol_node)

            # Convert each NH2 to NH3 (process in reverse to maintain indices)
            for n_idx in sorted(nh2_nitrogen_indices, reverse=True):
                try:
                    cleaned_atoms = convert_nh2_to_nh3(cleaned_atoms, n_idx)
                except ValueError:
                    # Skip if conversion fails
                    continue

        except Exception:
            # If pattern matching fails, skip conversion
            pass

    return cleaned_atoms
