"""GraphView class for graph-based molecule access and modification."""

from typing import List, Optional, Set, TYPE_CHECKING
import networkx as nx
from ase import Atoms

from q2D_Materials.analyzer.utils.pymatgen_utils import build_molecular_graph
from q2D_Materials.utils.molecules.molecule_builder import smiles_to_ase_atoms

from .molecule_graph import MoleculeGraph, MoleculesList

if TYPE_CHECKING:
    from q2D_Materials.analyzer import q2D_analyzer


class GraphView:
    """Graph-based view of a perovskite structure for molecule modification.

    Provides access to molecular fragments as MoleculeGraph objects with
    modification capabilities. Returns full ASE Atoms structures after modifications.

    Attributes
    ----------
    analyzer : q2D_analyzer
        The underlying analyzed structure
    full_structure : Atoms
        Reference to the complete original structure
    """

    def __init__(self, analyzer: "q2D_analyzer"):
        """Initialize GraphView from an analyzed structure.

        Parameters
        ----------
        analyzer : q2D_analyzer
            Analyzed q2D_analyzer instance (analyze() must have been called)

        Raises
        ------
        ValueError
            If analyzer has not been analyzed
        """
        if not analyzer._analyzed:
            raise ValueError(
                "Analyzer must be analyzed before creating GraphView. "
                "Call analyzer.analyze() first."
            )

        self.analyzer = analyzer
        self.full_structure = analyzer.cell
        self._molecules: Optional[List[MoleculeGraph]] = None
        self._spacers: Optional[List[MoleculeGraph]] = None
        self._a_sites: Optional[List[MoleculeGraph]] = None

    @property
    def molecules(self) -> MoleculesList:
        """Access all molecules (spacers + molecular A-sites)."""
        if self._molecules is None:
            self._molecules = self._extract_molecules()
        return MoleculesList(self._molecules)

    @property
    def spacers(self) -> MoleculesList:
        """Access only spacer molecules."""
        if self._spacers is None:
            self._extract_molecules()
        return MoleculesList(self._spacers or [])

    @property
    def a_sites(self) -> MoleculesList:
        """Access only molecular A-site cations."""
        if self._a_sites is None:
            self._extract_molecules()
        return MoleculesList(self._a_sites or [])

    def _extract_molecules(self) -> List[MoleculeGraph]:
        """Extract molecules from the analyzer as MoleculeGraph objects."""
        molecules = []
        self._spacers = []
        self._a_sites = []

        atoms_in_octahedra = self._get_atoms_in_octahedra()

        # Get spacers
        spacers = self.analyzer.get_spacers()

        for spacer in spacers:
            original_indices = spacer.info.get(
                'original_indices', list(range(len(spacer)))
            )
            attachment_points = spacer.info.get('attachment_atoms', [])

            # If attachment_atoms not in info, try to compute from graph
            if not attachment_points:
                attachment_points = self._find_attachment_points(
                    original_indices, atoms_in_octahedra
                )

            # Build molecular graph with original indices mapping
            spacer_graph = build_molecular_graph(
                spacer,
                exclude_indices=set(),
                original_indices=original_indices
            )

            mol_graph = MoleculeGraph(
                graph=spacer_graph,
                original_indices=original_indices,
                attachment_points=attachment_points,
                molecule_type='spacer',
                molecule_index=len(molecules),
                atoms_object=spacer,
                parent_view=self,
            )
            molecules.append(mol_graph)
            self._spacers.append(mol_graph)

        # Get A-sites (molecular ones)
        a_sites = self.analyzer.get_a_sites()
        for a_site in a_sites:
            if a_site.get('is_molecular', False):
                molecule_atoms = a_site.get('molecule_atoms', [])
                if len(molecule_atoms) > 1:
                    # Create Atoms object for A-site molecule
                    a_site_atoms = Atoms(
                        symbols=[self.full_structure[i].symbol for i in molecule_atoms],
                        positions=[self.full_structure[i].position for i in molecule_atoms],
                    )
                    a_site_atoms.info['original_indices'] = molecule_atoms

                    # Build graph with original indices mapping
                    a_site_graph = build_molecular_graph(
                        a_site_atoms,
                        exclude_indices=set(),
                        original_indices=molecule_atoms
                    )

                    mol_graph = MoleculeGraph(
                        graph=a_site_graph,
                        original_indices=molecule_atoms,
                        attachment_points=[],  # A-sites typically don't have framework attachments
                        molecule_type='a_site',
                        molecule_index=len(molecules),
                        atoms_object=a_site_atoms,
                        parent_view=self,
                    )
                    molecules.append(mol_graph)
                    self._a_sites.append(mol_graph)

        return molecules

    def _get_atoms_in_octahedra(self) -> Set[int]:
        """Get set of atom indices that are part of octahedra."""
        atoms_in_octahedra: Set[int] = set()

        for node, data in self.analyzer.get_graph().nodes(data=True):
            if data.get('node_type') == 'octahedron':
                central = data.get('central_atom')
                if central is not None:
                    atoms_in_octahedra.add(central)

                for atom_list in [
                    data.get('terminal_atoms', []),
                    data.get('interlayer_atoms', []),
                    data.get('intralayer_atoms', []),
                ]:
                    atoms_in_octahedra.update(atom_list)

        return atoms_in_octahedra

    def _find_attachment_points(
        self,
        molecule_indices: List[int],
        atoms_in_octahedra: Set[int],
    ) -> List[int]:
        """Find atoms that connect to the inorganic framework."""
        attachment_points = []
        graph = self.analyzer.get_graph()

        for idx in molecule_indices:
            atom_node = f'atom_{idx}'
            if not graph.has_node(atom_node):
                continue

            for neighbor in graph.neighbors(atom_node):
                if neighbor.startswith('atom_'):
                    neighbor_idx = int(neighbor.replace('atom_', ''))
                    if neighbor_idx in atoms_in_octahedra:
                        attachment_points.append(idx)
                        break

        return attachment_points

    def from_smiles(self, smiles: str) -> nx.Graph:
        """Convert SMILES string to NetworkX graph using graph-based parsing.

        Uses fast direct SMILES parsing without 3D coordinate generation.
        Graph matching is based on element symbols and connectivity only.

        Parameters
        ----------
        smiles : str
            SMILES string representing the fragment

        Returns
        -------
        nx.Graph
            NetworkX graph with atom indices as nodes and bonds as edges.
            Each node has 'symbol' attribute and optional attributes like
            'charge', 'hcount', 'aromatic', etc.
            No 3D positions - graph matching is connectivity-based.
        """
        from q2D_Materials.modifier.fragment import from_smiles
        return from_smiles(smiles, validate=False)

    def get_full_structure(self) -> Atoms:
        """Return a copy of the full original structure.

        Returns
        -------
        Atoms
            Copy of the complete structure
        """
        return self.full_structure.copy()
