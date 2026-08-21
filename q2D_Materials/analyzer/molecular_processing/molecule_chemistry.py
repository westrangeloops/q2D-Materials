"""Chemical completeness predicates for molecular graphs.

Thin checks over bonded_to neighbors and existing valence/radii tables.
No graph construction and no new chemical databases.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from ..utils.graph_utils import (
    filter_nodes_by_attributes,
    get_connected_components,
    get_node_neighbors,
)
from ...utils.properties.atomic_properties import (
    estimate_bond_order,
    get_valence,
)

# Degree must equal get_valence(symbol) unless overridden here.
DEGREE_ALLOWED: Dict[str, Set[int]] = {
    "N": {3, 4},  # table valence is 3; ammonium is 4
    "O": {1, 2},  # table valence is 2; carbonyl is 1
    "S": {2, 4, 6},
}

ORGANIC_HEAVY = frozenset({"C", "N", "O", "S", "P"})
ELEMENTS_TO_CHECK = ("H", "C", "N", "O", "S", "P", "F")


def bonded_neighbor_nodes(graph: nx.Graph, node: Any) -> List[Any]:
    """Atom neighbor node IDs via bonded_to edges."""
    nbrs = get_node_neighbors(graph, node, edge_type="bonded_to")
    return [
        n for n in nbrs
        if graph.nodes[n].get("node_type", "atom") == "atom"
        and "symbol" in graph.nodes[n]
    ]


def bonded_symbols(graph: nx.Graph, node: Any) -> List[str]:
    """Element symbols of bonded_to atom neighbors."""
    return [graph.nodes[n]["symbol"] for n in bonded_neighbor_nodes(graph, node)]


def bond_order_to_neighbor(graph: nx.Graph, node: Any, neighbor: Any) -> float:
    """Bond order from edge ``order``, else distance estimate, else 1."""
    edge = graph.get_edge_data(node, neighbor) or {}
    if "order" in edge and edge["order"] is not None:
        try:
            return float(edge["order"])
        except (TypeError, ValueError):
            pass
    distance = edge.get("distance")
    if distance is not None:
        return float(
            estimate_bond_order(
                float(distance),
                graph.nodes[node]["symbol"],
                graph.nodes[neighbor]["symbol"],
            )
        )
    return 1.0


def effective_valence(graph: nx.Graph, node: Any) -> float:
    """Sum of bond orders to atom neighbors."""
    return sum(
        bond_order_to_neighbor(graph, node, n)
        for n in bonded_neighbor_nodes(graph, node)
    )


def _atom_nodes(graph: nx.Graph) -> List[Any]:
    """Atom nodes in the molecular graph (exclude molecule_0 wrappers)."""
    atoms = filter_nodes_by_attributes(graph, node_type="atom")
    if atoms:
        return atoms
    # Flat atom graphs from converters may omit node_type
    return [
        n for n, d in graph.nodes(data=True)
        if "symbol" in d and not str(n).startswith(("molecule_", "spacer_", "a_site_"))
    ]


def _atom_subgraph(graph: nx.Graph) -> nx.Graph:
    atoms = _atom_nodes(graph)
    return graph.subgraph(atoms).copy()


def check_connected(graph: nx.Graph) -> Tuple[bool, str, str]:
    """Heavy atoms (and all atoms) must form a single bonded component."""
    atom_g = _atom_subgraph(graph)
    if atom_g.number_of_nodes() == 0:
        return True, "ok", ""
    # Restrict to bonded_to edges only
    bonded = nx.Graph()
    bonded.add_nodes_from(atom_g.nodes(data=True))
    for u, v, data in atom_g.edges(data=True):
        if data.get("edge_type", "bonded_to") == "bonded_to":
            bonded.add_edge(u, v, **data)
    components = get_connected_components(bonded)
    if len(components) > 1:
        return False, "disconnected", f"{len(components)} connected components"
    return True, "ok", ""


def check_organic_has_h(graph: nx.Graph) -> Tuple[bool, str, str]:
    """If the molecule has C or N, it must also have at least one H."""
    symbols = [graph.nodes[n].get("symbol") for n in _atom_nodes(graph)]
    has_cn = any(s in ("C", "N") for s in symbols)
    has_h = any(s == "H" for s in symbols)
    if has_cn and not has_h:
        return False, "missing_hydrogens", "organic molecule has C/N but no H atoms"
    return True, "ok", ""


def check_element_degrees(
    graph: nx.Graph,
    symbol: str,
    allowed: Optional[Set[int]] = None,
) -> Tuple[bool, str, str]:
    """Every atom of ``symbol`` must have coordination in the allowed set.

    For carbon:
    - If edges carry RDKit ``order``, require bond-order sum ≈ 4.
    - CIF distance-only graphs: degree 4 always OK; degree 3 OK for sp2
      (in a 5–7 ring / aromatic, or bonded to O as carboxyl/carbonyl).
      Alkyl CH2 missing one H (2 heavy + 1 H, not in a ring) still fails.
    """
    if symbol == "C":
        for node in _atom_nodes(graph):
            if graph.nodes[node].get("symbol") != "C":
                continue
            nbrs = bonded_neighbor_nodes(graph, node)
            degree = len(nbrs)
            has_explicit_order = any(
                (graph.get_edge_data(node, n) or {}).get("order") is not None
                for n in nbrs
            )
            if has_explicit_order:
                val = effective_valence(graph, node)
                if abs(val - 4.0) >= 0.6:
                    return (
                        False,
                        "bad_valence",
                        f"C atom {node} has effective valence {val:.2f} "
                        f"(degree {degree}), expected ~4",
                    )
                continue
            if degree == 4:
                continue
            if degree == 3 and _carbon_sp2_ok(graph, node, nbrs):
                continue
            return (
                False,
                "bad_valence",
                f"C atom {node} has degree {degree}, allowed [4] "
                f"(or sp2 ring/carboxyl degree 3)",
            )
        return True, "ok", ""

    if allowed is None:
        allowed = DEGREE_ALLOWED.get(symbol)
        if allowed is None:
            allowed = {get_valence(symbol)}

    for node in _atom_nodes(graph):
        if graph.nodes[node].get("symbol") != symbol:
            continue
        degree = len(bonded_symbols(graph, node))
        if degree not in allowed:
            return (
                False,
                "bad_valence",
                f"{symbol} atom {node} has degree {degree}, allowed {sorted(allowed)}",
            )
    return True, "ok", ""


def _carbon_sp2_ok(graph: nx.Graph, node: Any, nbrs: List[Any]) -> bool:
    """Degree-3 carbon allowed if carboxyl/carbonyl or in a small ring."""
    heavy = [n for n in nbrs if graph.nodes[n].get("symbol") != "H"]
    if any(graph.nodes[n].get("symbol") == "O" for n in heavy):
        return True
    return _atom_in_small_ring(graph, node, sizes=range(5, 8))


def _atom_in_small_ring(
    graph: nx.Graph,
    node: Any,
    sizes=range(5, 8),
) -> bool:
    """True if ``node`` lies on a heavy-atom cycle of allowed size."""
    size_set = set(sizes)
    atoms = _atom_nodes(graph)
    heavy = [n for n in atoms if graph.nodes[n].get("symbol") != "H"]
    if node not in heavy:
        return False
    sub = nx.Graph()
    sub.add_nodes_from(heavy)
    for u, v, data in graph.edges(data=True):
        if u in sub and v in sub and data.get("edge_type", "bonded_to") == "bonded_to":
            sub.add_edge(u, v)
    if node not in sub or sub.degree(node) < 2:
        return False
    for cycle in nx.cycle_basis(sub):
        if node in cycle and len(cycle) in size_set:
            return True
    return False


def check_max_valence(graph: nx.Graph) -> Tuple[bool, str, str]:
    """Degree must not exceed get_valence(symbol); N may be 4."""
    for node in _atom_nodes(graph):
        symbol = graph.nodes[node].get("symbol")
        if not symbol:
            continue
        degree = len(bonded_symbols(graph, node))
        max_v = get_valence(symbol)
        if symbol == "N":
            max_v = max(max_v, 4)
        if degree > max_v:
            return (
                False,
                "bad_valence",
                f"{symbol} atom {node} degree {degree} exceeds max valence {max_v}",
            )
    return True, "ok", ""


def check_oxygen_bond_order(graph: nx.Graph) -> Tuple[bool, str, str]:
    """O with one neighbor: double bond (carbonyl) passes; single bond fails."""
    for node in _atom_nodes(graph):
        if graph.nodes[node].get("symbol") != "O":
            continue
        nbrs = get_node_neighbors(graph, node, edge_type="bonded_to")
        atom_nbrs = [
            n for n in nbrs
            if graph.nodes[n].get("node_type", "atom") == "atom"
            and "symbol" in graph.nodes[n]
        ]
        if len(atom_nbrs) != 1:
            continue
        partner = atom_nbrs[0]
        edge = graph.get_edge_data(node, partner) or {}
        distance = edge.get("distance")
        if distance is None:
            # No distance: allow degree-1 O (treated as carbonyl-like)
            continue
        order = estimate_bond_order(
            float(distance),
            "O",
            graph.nodes[partner]["symbol"],
        )
        if order < 2:
            return (
                False,
                "missing_hydrogens",
                f"O atom {node} has single bond to {partner} (likely missing H)",
            )
    return True, "ok", ""


def check_rdkit_sanitize(graph: nx.Graph) -> Tuple[bool, str, str]:
    """Optional RDKit SanitizeMol on the isolated atom subgraph only."""
    try:
        from ...utils.molecules.graph_converter import graph_to_rdkit
        from rdkit import Chem
    except ImportError:
        return True, "ok", "rdkit unavailable"

    atom_g = _atom_subgraph(graph)
    # graph_to_rdkit needs integer-like sequential nodes with symbol
    try:
        # Relabel to 0..n-1 if needed
        mapping = {n: i for i, n in enumerate(sorted(atom_g.nodes(), key=str))}
        relabeled = nx.relabel_nodes(atom_g, mapping, copy=True)
        mol = graph_to_rdkit(relabeled, preserve_coords=False)
        Chem.SanitizeMol(mol)
    except Exception as e:
        return False, "sanitize_failed", str(e)
    return True, "ok", ""


def check_molecule_chemistry(
    graph: nx.Graph,
    *,
    sanitize: bool = False,
) -> Tuple[bool, str, str]:
    """Run stage-1 chemistry checks; stop at first failure.

    Returns
    -------
    (ok, reason_code, detail)
        reason_code is ``ok``, ``disconnected``, ``missing_hydrogens``,
        ``bad_valence``, or ``sanitize_failed``.
    """
    checks = [
        check_connected,
        check_organic_has_h,
        lambda g: _check_all_element_degrees(g),
        check_max_valence,
        check_oxygen_bond_order,
    ]
    if sanitize:
        checks.append(check_rdkit_sanitize)

    for check in checks:
        ok, reason, detail = check(graph)
        if not ok:
            return ok, reason, detail
    return True, "ok", ""


def _check_all_element_degrees(graph: nx.Graph) -> Tuple[bool, str, str]:
    present = {graph.nodes[n].get("symbol") for n in _atom_nodes(graph)}
    for symbol in ELEMENTS_TO_CHECK:
        if symbol not in present:
            continue
        ok, reason, detail = check_element_degrees(graph, symbol)
        if not ok:
            return ok, reason, detail
    return True, "ok", ""


def is_organic_molecule(graph: nx.Graph) -> bool:
    """True if the molecule contains C, N, or O."""
    for node in _atom_nodes(graph):
        if graph.nodes[node].get("symbol") in ("C", "N", "O"):
            return True
    return False
