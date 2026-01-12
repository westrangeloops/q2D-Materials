"""Molecule candidate analysis for DJ and RP spacer suitability.

This module provides functions for analyzing whether molecules are suitable as
DJ (Dion-Jacobson) or RP (Ruddlesden-Popper) spacers based on pattern matching
with user-defined SMILES patterns.
"""

from typing import Union, Optional, List, Tuple, Set, Dict
from dataclasses import dataclass
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from ase import Atoms
from collections import Counter

from ..utils.pymatgen_utils import build_molecular_graph
from ..utils.graph_utils import find_shortest_path, validate_path_continuity, get_connected_components
from ...utils.molecules.smiles_parser import smiles_to_graph as direct_smiles_to_graph
from ...builders.optimizers import elongate_molecule
from ...modifier.fragment import from_smiles, validate_fragment

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
DEFAULT_INITIAL_PATTERN: str = 'NH2C'  # NH2 bonded to carbon
DEFAULT_FINAL_PATTERN: str = 'NH2C'   # NH2 bonded to carbon

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

def _find_attachment_carbon(pattern_graph: nx.Graph) -> Optional[int]:
    """
    Identify carbon attachment point in SMILES pattern.

    Searches for carbon atom in the pattern graph. For patterns like 'NH2C',
    '[NH3+]C', returns the carbon node index.

    Parameters
    ----------
    pattern_graph : nx.Graph
        NetworkX graph from from_smiles()

    Returns
    -------
    Optional[int]
        Carbon node index, or None if no carbon found
    """
    for node in pattern_graph.nodes():
        symbol = pattern_graph.nodes[node].get('symbol', '')
        if symbol == 'C':
            return node
    return None


def _parse_pattern(pattern_smiles: str) -> Tuple[nx.Graph, Optional[int]]:
    """
    Convert SMILES pattern to graph and identify attachment carbon.

    Uses fast direct SMILES parsing (no 3D coordinates needed for pattern matching).

    Examples:
        '[NH3+]C' → graph with N-C bond, returns C index
        'NH2C' → graph with N-C bond, returns C index
        '[NH3+]' → graph with just N, returns None (no carbon)

    Parameters
    ----------
    pattern_smiles : str
        SMILES string defining the terminal pattern

    Returns
    -------
    Tuple[nx.Graph, Optional[int]]
        (pattern_graph, carbon_attachment_index)

    Raises
    ------
    ValueError
        If SMILES pattern is invalid
    """
    try:
        # Use fast direct SMILES parsing (no 3D coordinates needed)
        pattern_graph = direct_smiles_to_graph(pattern_smiles, add_positions=False)

        # Find the carbon atom (attachment point)
        carbon_idx = _find_attachment_carbon(pattern_graph)

        return pattern_graph, carbon_idx

    except Exception as e:
        raise ValueError(f"Invalid SMILES pattern '{pattern_smiles}': {e}")


def _find_pattern_matches(
    molecule_graph: nx.Graph,
    pattern_graph: nx.Graph,
    pattern_carbon_idx: Optional[int],
    atoms: Optional[Atoms] = None
) -> List[Dict]:
    """
    Find all subgraph matches using NetworkX isomorphism.

    Uses subgraph isomorphism to find all occurrences of the pattern
    in the molecule. Nodes are matched by chemical symbol from graph attributes.
    Graph-based matching is fast, accurate, and geometry-independent.

    Parameters
    ----------
    molecule_graph : nx.Graph
        Full molecular connectivity graph (graph-based, no 3D coordinates needed)
    pattern_graph : nx.Graph
        Pattern subgraph to match
    pattern_carbon_idx : Optional[int]
        Index of carbon in pattern (None if pattern has no carbon)
    atoms : Atoms, optional
        ASE Atoms object (deprecated - symbols now come from graph nodes)
        Kept for backward compatibility but not used

    Returns
    -------
    List[Dict]
        List of matches: [{'mapping': {pattern_idx: mol_idx}, 'carbon_idx': mol_carbon_idx}, ...]
        If pattern has no carbon, carbon_idx is None
    """
    # Define node matching function: match by chemical symbol from graph nodes
    def node_match(mol_node_data, pat_node_data):
        # Get symbol from graph node attributes (graph-based matching)
        mol_symbol = mol_node_data.get('symbol', '')
        pat_symbol = pat_node_data.get('symbol', '')
        
        # Also match charge if present (for patterns like [NH3+])
        mol_charge = mol_node_data.get('charge', 0)
        pat_charge = pat_node_data.get('charge', 0)
        
        # Match symbol and charge
        return mol_symbol == pat_symbol and mol_charge == pat_charge

    # Use NetworkX GraphMatcher for subgraph isomorphism
    matcher = isomorphism.GraphMatcher(
        molecule_graph,
        pattern_graph,
        node_match=node_match
    )

    matches = []
    for mapping in matcher.subgraph_isomorphisms_iter():
        # mapping: molecule_node → pattern_node (from NetworkX)
        # To find the molecule carbon, find which molecule node maps to pattern carbon
        mol_carbon_idx = None
        if pattern_carbon_idx is not None:
            # Find molecule node that maps to pattern carbon node
            for mol_node, pat_node in mapping.items():
                if pat_node == pattern_carbon_idx:
                    mol_carbon_idx = mol_node
                    break

        matches.append({
            'mapping': mapping,
            'carbon_idx': mol_carbon_idx
        })

    return matches


def _find_valid_paths_pattern_based(
    graph: nx.Graph,
    initial_matches: List[Dict],
    final_matches: List[Dict],
    min_length: int,
    atoms: Optional[Atoms] = None,
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

    for init_match in initial_matches:
        for final_match in final_matches:
            c1 = init_match['carbon_idx']
            c2 = final_match['carbon_idx']

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
    >>> result = analyzer.analyze_molecule_as_dj_spacer(
    ...     "NCCCCN",
    ...     initial_pattern='NH2C',
    ...     final_pattern='NH2C'
    ... )
    >>>
    >>> # Ammonium terminals
    >>> result = analyzer.analyze_molecule_as_dj_spacer(
    ...     molecule,
    ...     initial_pattern='[NH3+]C',
    ...     final_pattern='[NH3+]C'
    ... )
    >>>
    >>> # Multiple final patterns
    >>> result = analyzer.analyze_molecule_as_dj_spacer(
    ...     molecule,
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
            # Use direct SMILES parsing - fast, graph-based, no 3D coordinates
            graph = direct_smiles_to_graph(molecule, add_positions=False)
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
        # If Atoms object provided, build graph from it (for backward compatibility)
        atoms = molecule
        try:
            indices = list(range(len(atoms)))
            graph = build_molecular_graph(atoms, set(), indices)
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
            pat_graph, pat_carbon_idx = _parse_pattern(pat)
            matches = _find_pattern_matches(graph, pat_graph, pat_carbon_idx, None)
            all_initial_matches.extend(matches)

        # Find all final pattern matches
        all_final_matches = []
        for pat in final_patterns:
            pat_graph, pat_carbon_idx = _parse_pattern(pat)
            matches = _find_pattern_matches(graph, pat_graph, pat_carbon_idx, None)
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
            atoms,
            allowed_backbone_elements,
            forbidden_backbone_elements,
            max_non_carbon_ratio,
            allow_same_carbon
        )

        if len(valid_paths) == 0:
            # Check if we have distinct terminals (for better error message)
            initial_carbons = {m['carbon_idx'] for m in all_initial_matches if m['carbon_idx'] is not None}
            final_carbons = {m['carbon_idx'] for m in all_final_matches if m['carbon_idx'] is not None}
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
        matches_with_carbon = [m for m in all_initial_matches if m['carbon_idx'] is not None]
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

    # Build graph to find H neighbors
    indices = list(range(len(atoms)))
    graph = build_molecular_graph(atoms, set(), indices)

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

    # Build molecular graph
    indices = list(range(len(atoms)))
    graph = build_molecular_graph(atoms, set(), indices)

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
                    pat_graph, pat_carbon_idx = _parse_pattern(pattern)
                    matches = _find_pattern_matches(subgraph, pat_graph, pat_carbon_idx, atoms)
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
            # Rebuild graph for cleaned atoms
            indices = list(range(len(cleaned_atoms)))
            graph = build_molecular_graph(cleaned_atoms, set(), indices)

            # Search for NH2 groups using fast direct parsing
            pat_graph, pat_carbon_idx = _parse_pattern('NH2C')
            matches = _find_pattern_matches(graph, pat_graph, pat_carbon_idx, cleaned_atoms)

            # Find nitrogen indices from matches
            nh2_nitrogen_indices = set()
            for match in matches:
                # Find nitrogen node in the match mapping
                for pat_node, mol_node in match['mapping'].items():
                    if pat_graph.nodes[pat_node].get('symbol') == 'N':
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
