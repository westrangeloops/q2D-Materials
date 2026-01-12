"""Molecule Modifier - Graph-based molecular fragment modification.

This module provides tools for modifying molecular fragments in perovskite
structures (DJ, RP, monolayer). Returns full ASE Atoms objects for output.

Example
-------
>>> from q2D_Materials.analyzer import q2D_analyzer
>>> from q2D_Materials.modifier import GraphView, from_smiles
>>>
>>> analyzer = q2D_analyzer("structure.vasp")
>>> analyzer.analyze()
>>>
>>> view = GraphView(analyzer)
>>> molecules = view.molecules.list()
>>>
>>> # Replace atom 5 in first molecule with ethyl group
>>> fragment = from_smiles("CC")
>>> modified = molecules[0].replace(atom_index=5, fragment_graph=fragment)
>>>
>>> # Write using ASE standard methods
>>> modified.write("modified.vasp")
"""

from .graph_view import GraphView
from .molecule_graph import MoleculeGraph, MoleculesList
from .fragment import from_smiles, atoms_to_smiles, validate_fragment

__all__ = [
    'GraphView',
    'MoleculeGraph',
    'MoleculesList',
    'from_smiles',
    'atoms_to_smiles',
    'validate_fragment',
]
