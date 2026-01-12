"""
Direct SMILES to Graph Parser - Bypasses 3D coordinate generation.

This module provides fast SMILES parsing directly to NetworkX graphs without
requiring 3D coordinate generation or geometry optimization. This is essential
for pattern matching and graph-based queries where 3D structure is not needed.

For cases where 3D coordinates are needed, use the full smiles_to_ase_atoms()
function which includes RDKit-based coordinate generation.
"""

import enum
import logging
from typing import Optional, Tuple, List, Dict, Set
import networkx as nx
import numpy as np

from ...utils.properties.atomic_properties import get_covalent_radius

LOGGER = logging.getLogger(__name__)


@enum.unique
class TokenType(enum.Enum):
    """Possible SMILES token types"""
    ATOM = 1
    BOND_TYPE = 2
    BRANCH_START = 3
    BRANCH_END = 4
    RING_NUM = 5
    EZSTEREO = 6
    CHIRAL = 7


def _tokenize(smiles: str):
    """
    Iterates over a SMILES string, yielding tokens.

    Parameters
    ----------
    smiles : str
        The SMILES string to iterate over

    Yields
    ------
    tuple(TokenType, int, str)
        A tuple describing the type of token, its position, and the associated
        data.
    """
    organic_subset = 'B C N O P S F Cl Br I * b c n o s p'.split()
    smiles_iter = iter(smiles)
    token = ''
    idx = -1
    peek = None
    
    while True:
        idx += 1
        char = peek if peek else next(smiles_iter, '')
        peek = None
        if not char:
            break
            
        if char == '[':
            token = char
            for char in smiles_iter:
                token += char
                if char == ']':
                    break
            yield TokenType.ATOM, idx, token
        elif char in organic_subset:
            peek = next(smiles_iter, '')
            if char + peek in organic_subset:
                yield TokenType.ATOM, idx, char + peek
                peek = None
            elif peek == 'H':
                # Check for H-count notation like 'NH2', 'NH3'
                peek2 = next(smiles_iter, '')
                if peek2.isdigit():
                    # Consume H and digit as part of atom token
                    h_digit = peek2
                    peek3 = next(smiles_iter, '')
                    if peek3.isdigit():
                        h_digit += peek3
                        peek = peek3
                    else:
                        peek = peek3
                    yield TokenType.ATOM, idx, char + 'H' + h_digit
                else:
                    # Just 'H' after element - yield element, then H separately
                    yield TokenType.ATOM, idx, char
                    yield TokenType.ATOM, idx + 1, peek
                    peek = peek2
            else:
                yield TokenType.ATOM, idx, char
        elif char in '-=#$:.':
            yield TokenType.BOND_TYPE, idx, char
        elif char == '(':
            yield TokenType.BRANCH_START, idx, '('
        elif char == ')':
            yield TokenType.BRANCH_END, idx, ')'
        elif char == '%':
            # Two-digit ring number
            yield TokenType.RING_NUM, idx, int(next(smiles_iter, '') + next(smiles_iter, ''))
        elif char in '/\\':
            yield TokenType.EZSTEREO, idx, char
        elif char.isdigit():
            yield TokenType.RING_NUM, idx, int(char)


def _parse_atom_token(atom_str: str) -> Dict:
    """
    Parse an atom token string to extract element, charge, etc.
    
    Parameters
    ----------
    atom_str : str
        Atom token like 'C', '[NH3+]', '[CH0]', 'NH2', etc.
        
    Returns
    -------
    dict
        Dictionary with 'symbol', 'charge', 'hcount', 'isotope', 'class'
    """
    result = {
        'symbol': None,
        'charge': 0,
        'hcount': None,
        'isotope': None,
        'class': None,
        'aromatic': False
    }
    
    # Handle H-count notation like 'NH2', 'NH3' (not in brackets)
    if len(atom_str) >= 2 and atom_str[0] == 'H' and atom_str[1:].isdigit():
        result['symbol'] = 'H'
        result['hcount'] = 1  # H itself
        return result
    elif len(atom_str) >= 3 and atom_str[1] == 'H' and atom_str[2:].isdigit():
        # Pattern like 'NH2', 'NH3'
        result['symbol'] = atom_str[0]
        result['hcount'] = int(atom_str[2:])
        return result
    
    # Handle bracketed atoms like [NH3+], [CH0], [C@H]
    if atom_str.startswith('[') and atom_str.endswith(']'):
        atom_str = atom_str[1:-1]
        
        # Parse isotope (if present at start)
        if atom_str and atom_str[0].isdigit():
            digits = ''
            i = 0
            while i < len(atom_str) and atom_str[i].isdigit():
                digits += atom_str[i]
                i += 1
            if digits:
                result['isotope'] = int(digits)
                atom_str = atom_str[len(digits):]
        
        # Parse element (one or two characters)
        if len(atom_str) >= 2 and atom_str[:2] in ['Cl', 'Br']:
            result['symbol'] = atom_str[:2]
            atom_str = atom_str[2:]
        elif atom_str:
            result['symbol'] = atom_str[0]
            atom_str = atom_str[1:]
        
        # Parse remaining attributes
        # Charge: +, ++, +2, -, --, -2
        if '+' in atom_str or '-' in atom_str:
            charge_str = ''
            i = 0
            while i < len(atom_str):
                if atom_str[i] in '+-':
                    charge_str += atom_str[i]
                    i += 1
                    # Check for number after +/-
                    if i < len(atom_str) and atom_str[i].isdigit():
                        num = ''
                        while i < len(atom_str) and atom_str[i].isdigit():
                            num += atom_str[i]
                            i += 1
                        charge_str += num
                    break
                i += 1
            
            if charge_str:
                if charge_str.startswith('+'):
                    if len(charge_str) == 1:
                        result['charge'] = 1
                    elif charge_str == '++':
                        result['charge'] = 2
                    else:
                        result['charge'] = int(charge_str[1:]) if charge_str[1:].isdigit() else len(charge_str)
                elif charge_str.startswith('-'):
                    if len(charge_str) == 1:
                        result['charge'] = -1
                    elif charge_str == '--':
                        result['charge'] = -2
                    else:
                        result['charge'] = -int(charge_str[1:]) if charge_str[1:].isdigit() else -len(charge_str)
                atom_str = atom_str.replace(charge_str, '')
        
        # Parse H count: H, H2, H3, H0
        if 'H' in atom_str:
            h_idx = atom_str.index('H')
            h_str = 'H'
            i = h_idx + 1
            while i < len(atom_str) and atom_str[i].isdigit():
                h_str += atom_str[i]
                i += 1
            if h_str == 'H':
                result['hcount'] = 1
            elif h_str[1:].isdigit():
                result['hcount'] = int(h_str[1:])
            atom_str = atom_str.replace(h_str, '')
        
        # Parse class: :0, :1, etc.
        if ':' in atom_str:
            class_idx = atom_str.index(':')
            class_str = atom_str[class_idx:]
            if class_str[1:].isdigit():
                result['class'] = int(class_str[1:])
            atom_str = atom_str.replace(class_str, '')
        
        # Parse stereochemistry: @, @@
        if '@' in atom_str:
            result['chiral'] = atom_str.count('@')
            atom_str = atom_str.replace('@', '')
    else:
        # Simple atom (no brackets)
        if atom_str.lower() == atom_str:
            # Lowercase = aromatic
            result['symbol'] = atom_str.upper()
            result['aromatic'] = True
        else:
            result['symbol'] = atom_str
    
    return result


def smiles_to_graph(smiles: str, add_positions: bool = False) -> nx.Graph:
    """
    Parse SMILES string directly to NetworkX graph without 3D coordinate generation.
    
    This is much faster than smiles_to_ase_atoms() + build_molecular_graph() because
    it bypasses RDKit 3D embedding and UFF optimization.
    
    Parameters
    ----------
    smiles : str
        SMILES string to parse
    add_positions : bool, default=False
        If True, adds approximate 3D positions using covalent radii.
        If False, only creates connectivity graph (much faster).
        
    Returns
    -------
    nx.Graph
        NetworkX graph with:
        - Nodes: atom indices (0, 1, 2, ...)
        - Node attributes: 'symbol', 'charge', 'hcount', 'aromatic', etc.
        - Edges: covalent bonds
        - Edge attributes: 'order' (bond order: 1, 2, 3, 1.5 for aromatic)
        - Optional: 'position' (if add_positions=True)
        
    Examples
    --------
    >>> graph = smiles_to_graph("NCCCCN")
    >>> len(graph.nodes())  # 6 atoms
    6
    >>> graph.nodes[0]['symbol']  # 'N'
    'N'
    >>> graph.has_edge(0, 1)  # N-C bond
    True
    """
    bond_to_order = {'-': 1, '=': 2, '#': 3, '$': 4, ':': 1.5, '.': 0}
    default_bond = 1
    default_aromatic_bond = 1.5
    
    mol = nx.Graph()
    mol.graph['smiles'] = smiles
    
    anchor = None
    idx = 0
    next_bond = None
    branches = []
    ring_nums = {}
    
    # First pass: parse structure
    for tokentype, token_idx, token in _tokenize(smiles):
        if tokentype == TokenType.ATOM:
            # Parse atom
            atom_data = _parse_atom_token(token)
            symbol = atom_data['symbol']
            
            if symbol is None:
                raise ValueError(f"Could not parse atom from token: {token}")
            
            # Add node
            mol.add_node(idx, **atom_data, _pos=token_idx, _atom_str=token)
            
            # Add edge to previous atom
            if anchor is not None:
                if next_bond is None:
                    next_bond = ""
                bond_order = bond_to_order.get(next_bond, default_bond)
                mol.add_edge(anchor, idx, order=bond_order, _pos=token_idx, _bond_str=next_bond)
                next_bond = None
            
            anchor = idx
            idx += 1
            
        elif tokentype == TokenType.BRANCH_START:
            if anchor is None:
                raise SyntaxError('Cannot start a branch before an atom.')
            branches.append(anchor)
            
        elif tokentype == TokenType.BRANCH_END:
            if not branches:
                raise SyntaxError('Unmatched closing parenthesis.')
            anchor = branches.pop()
            
        elif tokentype == TokenType.BOND_TYPE:
            if next_bond is not None:
                raise ValueError(f'Previous bond ({next_bond}) not used. Overwritten by "{token}"')
            next_bond = token
            
        elif tokentype == TokenType.RING_NUM:
            if token in ring_nums:
                # Close ring
                jdx, order = ring_nums[token]
                if next_bond is None and order is None:
                    next_bond = ""
                elif order is None:
                    pass  # Use next_bond as is
                elif next_bond is None:
                    next_bond = order
                elif next_bond != order:
                    raise ValueError(f'Conflicting bond orders for ring marker {token}')
                
                if mol.has_edge(idx - 1, jdx):
                    raise ValueError(f'Edge specified by marker {token} already exists')
                if idx - 1 == jdx:
                    raise ValueError(f'Marker {token} specifies bond to self')
                
                bond_order = bond_to_order.get(next_bond, default_bond)
                mol.add_edge(idx - 1, jdx, order=bond_order, _pos=token_idx, _bond_str=next_bond)
                next_bond = None
                del ring_nums[token]
            else:
                # Open ring
                if idx == 0:
                    raise ValueError(f"Can't have ring marker ({token}) before an atom")
                ring_nums[token] = (idx - 1, next_bond)
                next_bond = None
    
    # Allow unmatched ring indices for pattern matching (be lenient)
    # This is useful for partial SMILES patterns like 'NH2C'
    if ring_nums:
        # Log warning but don't fail - useful for pattern matching
        LOGGER.debug(f'Unmatched ring indices (ignored for pattern matching): {list(ring_nums.keys())}')
    
    # Second pass: handle aromaticity and add positions if requested
    # For aromatic bonds, set order to 1.5 if not explicitly set
    for edge in mol.edges():
        if 'order' not in mol.edges[edge]:
            u, v = edge
            if mol.nodes[u].get('aromatic', False) and mol.nodes[v].get('aromatic', False):
                mol.edges[edge]['order'] = default_aromatic_bond
            else:
                mol.edges[edge]['order'] = default_bond
    
    # Add approximate positions using covalent radii if requested
    if add_positions:
        _add_approximate_positions(mol)
    
    # Clean up temporary attributes
    for node in mol.nodes():
        mol.nodes[node].pop('_atom_str', None)
        mol.nodes[node].pop('_pos', None)
    for edge in mol.edges():
        mol.edges[edge].pop('_bond_str', None)
        mol.edges[edge].pop('_pos', None)
    
    return mol


def _add_approximate_positions(graph: nx.Graph):
    """
    Add approximate 3D positions to graph nodes using covalent radii.
    
    Uses a simple distance-based layout where bond lengths are approximated
    as the sum of covalent radii.
    
    Parameters
    ----------
    graph : nx.Graph
        Graph to add positions to (modified in place)
    """
    if len(graph.nodes()) == 0:
        return
    
    # Simple layout: place atoms in a chain/ring structure
    # For more complex molecules, this is approximate
    positions = {}
    
    # Try to find a simple path through the molecule
    nodes = list(graph.nodes())
    if len(nodes) == 1:
        positions[nodes[0]] = np.array([0.0, 0.0, 0.0])
    else:
        # Use a simple BFS-based layout
        visited = set()
        queue = [(nodes[0], np.array([0.0, 0.0, 0.0]))]
        visited.add(nodes[0])
        positions[nodes[0]] = np.array([0.0, 0.0, 0.0])
        
        while queue:
            current, pos = queue.pop(0)
            symbol = graph.nodes[current].get('symbol', 'C')
            radius = get_covalent_radius(symbol)
            
            # Get neighbors not yet positioned
            neighbors = [n for n in graph.neighbors(current) if n not in visited]
            
            if neighbors:
                # Simple placement: place neighbors in a circle around current
                angle_step = 2 * np.pi / len(neighbors)
                for i, neighbor in enumerate(neighbors):
                    neighbor_symbol = graph.nodes[neighbor].get('symbol', 'C')
                    neighbor_radius = get_covalent_radius(neighbor_symbol)
                    bond_length = radius + neighbor_radius
                    
                    # Get bond order to adjust length
                    bond_order = graph.edges[current, neighbor].get('order', 1)
                    if bond_order == 2:
                        bond_length *= 0.9  # Double bonds are shorter
                    elif bond_order == 3:
                        bond_length *= 0.85  # Triple bonds are shorter
                    elif bond_order == 1.5:
                        bond_length *= 0.95  # Aromatic bonds
                    
                    angle = i * angle_step
                    direction = np.array([np.cos(angle), np.sin(angle), 0.0])
                    neighbor_pos = pos + direction * bond_length
                    
                    positions[neighbor] = neighbor_pos
                    visited.add(neighbor)
                    queue.append((neighbor, neighbor_pos))
    
    # Add positions to graph
    for node, pos in positions.items():
        graph.nodes[node]['position'] = pos
        graph.nodes[node]['original_index'] = node


def graph_to_smiles(graph: nx.Graph) -> str:
    """
    Convert a graph back to SMILES string (inverse of smiles_to_graph).
    
    This is a simplified implementation that works for simple molecules.
    For complex molecules with stereochemistry, use RDKit.
    
    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph with 'symbol' node attributes and 'order' edge attributes
        
    Returns
    -------
    str
        SMILES string representation
    """
    if len(graph.nodes()) == 0:
        return ""
    
    # Simple DFS-based SMILES generation
    visited = set()
    smiles_parts = []
    
    def dfs(node, parent=None):
        if node in visited:
            return
        visited.add(node)
        
        symbol = graph.nodes[node].get('symbol', 'C')
        aromatic = graph.nodes[node].get('aromatic', False)
        
        # Format atom
        if aromatic and symbol.isupper():
            atom_str = symbol.lower()
        elif symbol not in ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'] or \
             graph.nodes[node].get('charge', 0) != 0 or \
             graph.nodes[node].get('hcount') is not None:
            # Need brackets
            atom_str = f"[{symbol}"
            if graph.nodes[node].get('charge', 0) > 0:
                charge = graph.nodes[node]['charge']
                atom_str += '+' * charge if charge <= 2 else f"+{charge}"
            elif graph.nodes[node].get('charge', 0) < 0:
                charge = abs(graph.nodes[node]['charge'])
                atom_str += '-' * charge if charge <= 2 else f"-{charge}"
            if graph.nodes[node].get('hcount') is not None:
                hcount = graph.nodes[node]['hcount']
                atom_str += f"H{hcount}" if hcount > 1 else "H"
            atom_str += "]"
        else:
            atom_str = symbol.lower() if aromatic else symbol
        
        smiles_parts.append(atom_str)
        
        # Process neighbors
        neighbors = [n for n in graph.neighbors(node) if n != parent]
        if len(neighbors) > 1:
            # Branch
            for i, neighbor in enumerate(neighbors):
                if i > 0:
                    smiles_parts.append('(')
                bond_order = graph.edges[node, neighbor].get('order', 1)
                if bond_order == 2:
                    smiles_parts.append('=')
                elif bond_order == 3:
                    smiles_parts.append('#')
                dfs(neighbor, node)
                if i > 0:
                    smiles_parts.append(')')
        elif len(neighbors) == 1:
            bond_order = graph.edges[node, neighbors[0]].get('order', 1)
            if bond_order == 2:
                smiles_parts.append('=')
            elif bond_order == 3:
                smiles_parts.append('#')
            dfs(neighbors[0], node)
    
    # Start from a node with few neighbors (likely terminal)
    start_node = min(graph.nodes(), key=lambda n: len(list(graph.neighbors(n))))
    dfs(start_node)
    
    return ''.join(smiles_parts)

