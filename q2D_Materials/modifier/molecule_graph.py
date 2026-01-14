"""MoleculeGraph class for molecular fragment representation and modification."""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import List, Dict, Optional, TYPE_CHECKING
from ase import Atoms

from .fragment import atoms_to_smiles
from .structure_reconstruction import (
    reconstruct_structure,
    calculate_fragment_positions_geometry_aware
)
from q2D_Materials.utils.properties.atomic_properties import (
    validate_replacement,
    classify_neighbors,
    get_valence
)
from q2D_Materials.utils.molecules.graph_converter import (
    graph_to_rdkit,
    rdkit_to_graph,
    map_coordinates,
    transfer_coordinates,
    validate_conserved_atoms
)

try:
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdDistGeom
    RDKIT_EMBEDDING_AVAILABLE = True
except ImportError:
    RDKIT_EMBEDDING_AVAILABLE = False
    AllChem = None
    rdDistGeom = None

if TYPE_CHECKING:
    from .graph_view import GraphView


class MoleculeGraph:
    """Represents a molecular fragment as a NetworkX subgraph.

    Attributes
    ----------
    graph : nx.Graph
        NetworkX subgraph of the molecule
    original_indices : List[int]
        Atom indices in the original structure
    attachment_points : List[int]
        Atoms connected to the inorganic framework
    molecule_type : str
        'spacer' or 'a_site'
    molecule_index : int
        Index of this molecule in the list
    atoms_object : Atoms
        ASE Atoms object for the molecule
    parent_view : GraphView
        Reference to parent GraphView for full structure access
    """

    def __init__(
        self,
        graph: nx.Graph,
        original_indices: List[int],
        attachment_points: List[int],
        molecule_type: str,
        molecule_index: int,
        atoms_object: Atoms,
        parent_view: 'GraphView',
    ):
        self.graph = graph
        self.original_indices = original_indices
        self.attachment_points = attachment_points
        self.molecule_type = molecule_type
        self.molecule_index = molecule_index
        self.atoms_object = atoms_object
        self.parent_view = parent_view
        self._smiles: Optional[str] = None

    @property
    def smiles(self) -> Optional[str]:
        """Compute SMILES representation of the molecule."""
        if self._smiles is None:
            self._smiles = atoms_to_smiles(self.atoms_object)
        return self._smiles

    def to_json(self) -> Dict:
        """Return JSON representation with molecule index, SMILES, and atom info.

        Returns
        -------
        dict
            JSON-serializable dictionary with molecule information
        """
        atoms_info = []

        for local_idx, orig_idx in enumerate(self.original_indices):
            node_data = self.graph.nodes.get(orig_idx, {})
            symbol = node_data.get('symbol', self.atoms_object[local_idx].symbol)

            # Get neighbors in the molecule graph
            neighbors = list(self.graph.neighbors(orig_idx))
            local_neighbors = [
                self.original_indices.index(n)
                for n in neighbors
                if n in self.original_indices
            ]

            atoms_info.append({
                "index": local_idx,
                "original_index": orig_idx,
                "symbol": symbol,
                "neighbors": local_neighbors,
                "is_attachment": orig_idx in self.attachment_points,
            })

        return {
            "molecule_index": self.molecule_index,
            "smiles": self.smiles,
            "molecule_type": self.molecule_type,
            "n_atoms": len(self.original_indices),
            "attachment_points": [
                self.original_indices.index(ap)
                for ap in self.attachment_points
                if ap in self.original_indices
            ],
            "atoms": atoms_info,
        }

    def to_rdkit(self):
        """Convert to RDKit Mol for chemistry operations.
        
        Returns
        -------
        rdkit.Chem.Mol
            RDKit molecule object with coordinates preserved
        """
        return graph_to_rdkit(self.graph, preserve_coords=True)
    
    @classmethod
    def from_rdkit(
        cls,
        mol,
        original_indices: List[int],
        attachment_points: List[int],
        molecule_type: str,
        molecule_index: int,
        parent_view: 'GraphView',
        coords: Optional[np.ndarray] = None
    ):
        """Create MoleculeGraph from RDKit Mol.
        
        Parameters
        ----------
        mol : rdkit.Chem.Mol
            RDKit molecule object
        original_indices : List[int]
            Atom indices in the original structure
        attachment_points : List[int]
            Atoms connected to the inorganic framework
        molecule_type : str
            'spacer' or 'a_site'
        molecule_index : int
            Index of this molecule in the list
        parent_view : GraphView
            Reference to parent GraphView
        coords : np.ndarray, optional
            Alternative coordinate source
            
        Returns
        -------
        MoleculeGraph
            New MoleculeGraph instance
        """
        graph = rdkit_to_graph(mol, coords=coords)
        
        # Convert graph to Atoms for atoms_object
        from q2D_Materials.utils.molecules.graph_converter import graph_to_atoms
        atoms_object = graph_to_atoms(graph, validate_geometry=False)
        
        return cls(
            graph=graph,
            original_indices=original_indices,
            attachment_points=attachment_points,
            molecule_type=molecule_type,
            molecule_index=molecule_index,
            atoms_object=atoms_object,
            parent_view=parent_view
        )
    
    def update_from_rdkit(self, mol, validate: bool = True) -> Atoms:
        """Update molecule from modified RDKit Mol, preserve coordinates of unchanged atoms.
        
        Only new atoms/fragments are recalculated. Unchanged atoms preserve original coordinates.
        Requires at least 2 atoms to be conserved for reconstruction.
        
        Parameters
        ----------
        mol : rdkit.Chem.Mol
            Modified RDKit molecule object (should have explicit hydrogens via AddHs())
        validate : bool, default=True
            Validate that at least 2 atoms are conserved (always True, kept for API compatibility)
            
        Returns
        -------
        Atoms
            Complete modified structure (all atoms, not just this molecule)
            
        Raises
        ------
        ValueError
            If less than 2 atoms are conserved
            
        Notes
        -----
        The RDKit molecule should have explicit hydrogens added (via AddHs()) before
        calling this method to ensure all atoms, including terminal groups, are present.
        """
        # Ensure explicit hydrogens are present (RDKit uses implicit by default)
        # But first sanitize if needed
        from rdkit import Chem
        from rdkit.Chem import rdmolops
        
        if mol.GetNumAtoms() > 0:
            # Check if hydrogens are explicit FIRST (before sanitization)
            has_explicit_h = any(atom.GetSymbol() == 'H' for atom in mol.GetAtoms())
            
            if not has_explicit_h:
                # Sanitize first (required before AddHs)
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    try:
                        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                    except Exception:
                        pass  # Continue anyway
                
                # Add explicit hydrogens if not present
                mol = rdmolops.AddHs(mol)
            else:
                # Already has explicit hydrogens, just sanitize
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    try:
                        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                    except Exception:
                        pass  # Continue anyway
        
        new_graph = rdkit_to_graph(mol)
        
        # Map coordinates from old to new - detects only changed atoms
        mapping = map_coordinates(self.graph, new_graph)
        
        # Validate: at least 2 atoms must be conserved
        validate_conserved_atoms(mapping, min_conserved=2)
        
        # Transfer coordinates for conserved atoms
        new_graph = transfer_coordinates(self.graph, new_graph, mapping)
        
        # Generate coordinates for new atoms that don't have them
        # New atoms (not in mapping) need coordinates generated
        atoms_without_coords = []
        for node_idx in new_graph.nodes():
            if new_graph.nodes[node_idx].get('position') is None:
                atoms_without_coords.append(node_idx)
        
        if atoms_without_coords:
            # Generate 3D coordinates for new atoms using RDKit embedding
            if RDKIT_EMBEDDING_AVAILABLE:
                # Convert graph back to RDKit for coordinate generation
                # First, we need to preserve existing coordinates in the RDKit mol
                temp_mol = graph_to_rdkit(new_graph, preserve_coords=True)
                
                # Check if we already have a conformer (from preserved coords)
                if temp_mol.GetNumConformers() == 0:
                    # No conformer yet, generate one
                    try:
                        AllChem.EmbedMolecule(temp_mol, randomSeed=42)
                        try:
                            AllChem.MMFFOptimizeMolecule(temp_mol)
                        except Exception:
                            pass  # MMFF failed, but embedding succeeded
                    except Exception:
                        # Fallback to distance geometry
                        try:
                            rdDistGeom.EmbedMolecule(temp_mol)
                        except Exception:
                            pass  # Continue anyway
                else:
                    # We have a conformer, but need to optimize positions for new atoms
                    # RDKit will handle this automatically when we update the conformer
                    try:
                        AllChem.MMFFOptimizeMolecule(temp_mol)
                    except Exception:
                        pass  # Optimization failed, but conformer exists
                
                # Convert back to graph with coordinates (preserves existing, adds new)
                new_graph = rdkit_to_graph(temp_mol, coords=None)
        
        # Convert back to Atoms (RDKit structure already validated)
        from q2D_Materials.utils.molecules.graph_converter import graph_to_atoms
        new_atoms = graph_to_atoms(new_graph, validate_geometry=False)
        
        # Update internal state
        self.graph = new_graph
        self.atoms_object = new_atoms
        self._smiles = None  # Reset cached SMILES
        
        # Reconstruct full structure
        return reconstruct_structure(
            original_structure=self.parent_view.full_structure,
            old_molecule_indices=self.original_indices,
            new_molecule_atoms=new_atoms
        )

    def replace(
        self,
        atom_index: int,
        fragment_graph: nx.Graph,
        fragment_index: int = 0,
    ) -> Atoms:
        """Replace an atom in the molecule with a fragment, returning full structure.

        This method implements valence-aware replacement with the following rules:
        - Heavy atoms (C, N, O, etc.) in the structural skeleton are preserved
        - Hydrogens can be removed if the new atom has lower valence
        - Replacement is rejected if too many heavy atoms are connected
        
        IMPORTANT: Fragments must be unambiguous with explicit attachment points.
        Use SMILES notation like [CH0]=C for vinyl group (explicit attachment point).

        Parameters
        ----------
        atom_index : int
            Local index of the atom to replace in this molecule
        fragment_graph : nx.Graph
            NetworkX graph of the fragment to insert (from from_smiles())
        fragment_index : int, default=0
            Index of the atom in the fragment to use as attachment point

        Returns
        -------
        Atoms
            Complete modified structure (all atoms, not just this molecule).
            Can be written directly with ase.io.write() or atoms.write().

        Raises
        ------
        ValueError
            If replacement is chemically incompatible, fragment is ambiguous,
            or validation fails at any stage
        """
        # ====================================================================
        # STEP 1: VALIDATE INPUTS (before any processing)
        # ====================================================================
        
        # Validate atom_index
        if atom_index >= len(self.original_indices):
            raise ValueError(
                f"Atom index {atom_index} out of range "
                f"(molecule has {len(self.original_indices)} atoms)"
            )

        # Validate fragment_graph
        fragment_nodes = list(fragment_graph.nodes())
        if len(fragment_nodes) == 0:
            raise ValueError("Fragment graph is empty")
        
        if fragment_index >= len(fragment_nodes):
            raise ValueError(
                f"Fragment index {fragment_index} out of range "
                f"(fragment has {len(fragment_nodes)} atoms, indices 0-{len(fragment_nodes)-1})"
            )

        # Get fragment attachment atom info
        attachment_node = fragment_nodes[fragment_index]
        fragment_attach_symbol = fragment_graph.nodes[attachment_node].get('symbol', 'C')
        
        # Validate attachment point is reasonable (has at least one bond)
        fragment_attachment_neighbors = list(fragment_graph.neighbors(attachment_node))
        if len(fragment_attachment_neighbors) == 0:
            raise ValueError(
                f"Fragment attachment atom at index {fragment_index} ({fragment_attach_symbol}) "
                f"has no bonds. This is not a valid attachment point."
            )

        # ====================================================================
        # STEP 2: GET TARGET ATOM INFORMATION
        # ====================================================================
        
        # Get the original index of the atom to replace
        target_orig_idx = self.original_indices[atom_index]

        # Get target atom information
        target_node_data = self.graph.nodes[target_orig_idx]
        target_symbol = target_node_data.get('symbol', 'C')

        # Get neighbors of the target atom in the molecule graph
        target_neighbors = [
            n for n in self.graph.neighbors(target_orig_idx)
            if n in self.original_indices
        ]

        # Get neighbor symbols for validation
        neighbor_symbols = [
            self.graph.nodes[n].get('symbol', 'C')
            for n in target_neighbors
        ]

        # ====================================================================
        # STEP 3: VALIDATE COMPATIBILITY (before any processing)
        # ====================================================================
        
        # Validate replacement using valence rules
        is_valid, message = validate_replacement(
            old_element=target_symbol,
            new_element=fragment_attach_symbol,
            neighbor_symbols=neighbor_symbols
        )

        if not is_valid:
            raise ValueError(
                f"Invalid replacement: {message}\n"
                f"Target atom: {target_symbol} (local index {atom_index})\n"
                f"Fragment attachment: {fragment_attach_symbol} (fragment index {fragment_index})\n"
                f"Neighbors: {neighbor_symbols}"
            )

        # ====================================================================
        # STEP 4: PROCEED WITH REPLACEMENT (all validations passed)
        # ====================================================================
        
        # Count bonds the attachment atom has within the fragment

        # Separate into heavy and H bonds within fragment
        num_fragment_heavy_bonds = sum(
            1 for n in fragment_attachment_neighbors
            if fragment_graph.nodes[n].get('symbol') != 'H'
        )
        num_fragment_h = sum(
            1 for n in fragment_attachment_neighbors
            if fragment_graph.nodes[n].get('symbol') == 'H'
        )

        # Classify neighbors from structure
        n_heavy, n_hydrogens = classify_neighbors(neighbor_symbols)
        new_valence = get_valence(fragment_attach_symbol)

        # Calculate total heavy atom bonds
        total_heavy_bonds = num_fragment_heavy_bonds + n_heavy
        
        # Determine effective valence from connectivity (coordination number)
        # The effective valence IS the coordination number (num_neighbors) after replacement
        # This is determined by the actual connectivity, not hardcoded element-specific rules
        # The coordination number comes from the graph connectivity, which is stored in the graph nodes
        total_available_h = num_fragment_h + n_hydrogens
        
        # Get base valence from JSON file (source of truth)
        base_valence = new_valence  # This comes from atomic_valence.json
        
        # Calculate what the final coordination number will be
        # We need to find the coordination number that can accommodate:
        # - All heavy bonds (required)
        # - As many H as possible (up to available)
        # The coordination number = total_heavy_bonds + number_of_H_kept
        # number_of_H_kept = min(available_H, coordination - total_heavy_bonds)
        
        # Try coordination numbers starting from base_valence
        # The coordination number is the effective valence
        max_possible_coordination = min(total_heavy_bonds + total_available_h, 6)  # Reasonable upper limit
        
        # Get fragment attachment atom's connectivity from graph (connectivity-based, not hardcoded)
        # Use only heavy neighbors for expected coordination hint (H may be removed)
        fragment_attach_num_neighbors = len(fragment_attachment_neighbors)
        fragment_attach_hybridization = fragment_graph.nodes[attachment_node].get('hybridization', 'unknown')
        
        # Calculate expected coordination from connectivity (heavy bonds only, H is variable)
        # Expected coordination hint: fragment heavy neighbors + structure heavy neighbors
        # This gives us the minimum coordination needed (H can be added up to available)
        expected_coordination_hint = num_fragment_heavy_bonds + n_heavy
        
        # Find valid coordinations based on connectivity (not hardcoded element rules)
        # Coordination is valid if: coordination == total_heavy_bonds + min(h_slots, available_H)
        valid_coordinations = []
        for coordination in range(base_valence, max_possible_coordination + 1):
            h_slots = coordination - total_heavy_bonds
            if h_slots < 0:
                continue  # Can't accommodate all heavy bonds
            
            h_kept = min(h_slots, total_available_h)
            final_coordination = total_heavy_bonds + h_kept
            
            # Valid if coordination matches what we'd achieve
            if final_coordination == coordination:
                valid_coordinations.append(coordination)
        
        # Choose coordination based on connectivity (H availability)
        # This is connectivity-based: uses actual H counts from graph, not hardcoded thresholds
        if len(valid_coordinations) > 0:
            if total_available_h > 0:
                # When H is available, prefer coordination that uses H efficiently
                # Find coordination that can use at least 2 H when 5+ H available,
                # or at least 1 H otherwise
                min_h_to_use = 2 if total_available_h >= 5 else 1
                
                for coord in valid_coordinations:
                    h_slots = coord - total_heavy_bonds
                    if h_slots >= min_h_to_use:
                        effective_valence = coord
                        break
                else:
                    # No coordination meets H requirement, use minimum that can use some H
                    for coord in valid_coordinations:
                        if (coord - total_heavy_bonds) >= 1:
                            effective_valence = coord
                            break
                    else:
                        effective_valence = valid_coordinations[0]  # Fallback to minimum
            else:
                effective_valence = valid_coordinations[0]  # No H available, use minimum
        else:
            effective_valence = base_valence

        if total_heavy_bonds > effective_valence:
            raise ValueError(
                f"Replacement impossible: {fragment_attach_symbol} (effective valence {effective_valence}) "
                f"needs {total_heavy_bonds} bonds "
                f"({num_fragment_heavy_bonds} from fragment + {n_heavy} from structure). "
                f"Cannot accommodate all heavy atoms!"
            )

        # Calculate hydrogen slots available using effective valence
        h_slots_available = effective_valence - total_heavy_bonds

        # Determine which structure neighbors to keep
        # Strategy: Keep all heavy atoms, distribute H slots between structure and fragment
        neighbors_to_keep = []
        structure_h_to_keep = []

        # Keep all heavy atoms from structure
        for neighbor_idx in target_neighbors:
            neighbor_symbol = self.graph.nodes[neighbor_idx].get('symbol', 'C')
            if neighbor_symbol != 'H':
                neighbors_to_keep.append(neighbor_idx)
            else:
                structure_h_to_keep.append(neighbor_idx)

        # Decide how to distribute H slots
        # IMPORTANT: If fragment explicitly has 0 hydrogens (e.g., [CH0]), 
        # do NOT keep any structure hydrogens - respect the explicit specification
        h_from_fragment_to_keep = min(h_slots_available, num_fragment_h)
        
        # If fragment has 0 H, don't keep any structure H (explicit 0 H specification)
        # Otherwise, keep structure H only if there are extra slots after fragment H
        if num_fragment_h == 0:
            h_from_structure_to_keep = 0
        else:
            h_from_structure_to_keep = h_slots_available - h_from_fragment_to_keep

        # Keep the decided number of H from structure
        neighbors_to_keep.extend(structure_h_to_keep[:h_from_structure_to_keep])

        # Calculate what will be removed
        removed_h_structure = n_hydrogens - h_from_structure_to_keep
        removed_h_fragment = num_fragment_h - h_from_fragment_to_keep

        # Build modified molecule graph
        new_mol_graph = self.graph.copy()
        new_mol_graph.remove_node(target_orig_idx)

        # Remove hydrogen neighbors that weren't kept
        hydrogens_to_remove = [
            n for n in target_neighbors
            if n not in neighbors_to_keep and self.graph.nodes[n].get('symbol') == 'H'
        ]

        for h_idx in hydrogens_to_remove:
            if h_idx in new_mol_graph.nodes:
                new_mol_graph.remove_node(h_idx)

        # Map fragment indices to new indices (offset by max original index)
        max_orig_idx = max(self.original_indices) if self.original_indices else -1
        fragment_to_new = {}
        new_original_indices = [
            idx for idx in self.original_indices
            if idx != target_orig_idx and idx not in hydrogens_to_remove
        ]

        for frag_idx in fragment_nodes:
            new_idx = max_orig_idx + frag_idx + 1
            fragment_to_new[frag_idx] = new_idx
            new_original_indices.append(new_idx)

            # Copy node data
            node_data = fragment_graph.nodes[frag_idx].copy()
            new_mol_graph.add_node(new_idx, **node_data)

        # Add fragment edges
        for u, v, data in fragment_graph.edges(data=True):
            new_u = fragment_to_new[u]
            new_v = fragment_to_new[v]
            new_mol_graph.add_edge(new_u, new_v, **data)

        # Connect fragment attachment point to kept neighbors only
        attachment_new_idx = fragment_to_new[attachment_node]
        for neighbor_orig_idx in neighbors_to_keep:
            if neighbor_orig_idx in new_mol_graph.nodes:
                bond_data = {}
                if self.graph.has_edge(target_orig_idx, neighbor_orig_idx):
                    bond_data = self.graph.edges[target_orig_idx, neighbor_orig_idx].copy()
                new_mol_graph.add_edge(attachment_new_idx, neighbor_orig_idx, **bond_data)

        # Build new molecule Atoms object
        # Get positions for kept atoms (excluding replaced atom and removed hydrogens)
        new_positions = []
        new_symbols = []

        for local_idx, orig_idx in enumerate(self.original_indices):
            if orig_idx != target_orig_idx and orig_idx not in hydrogens_to_remove:
                new_positions.append(self.atoms_object[local_idx].position.copy())
                new_symbols.append(self.atoms_object[local_idx].symbol)

        # Get fragment atoms and filter out excess hydrogens
        fragment_atoms_full = self._graph_to_atoms(fragment_graph)

        # Map fragment graph node index to atoms array index
        # _graph_to_atoms uses sorted(graph.nodes()), so we need to map accordingly
        sorted_fragment_nodes = sorted(fragment_graph.nodes())
        fragment_node_to_atoms_idx = {node: idx for idx, node in enumerate(sorted_fragment_nodes)}
        
        # Get the attachment atom index in the full fragment_atoms array
        fragment_attachment_atoms_idx = fragment_node_to_atoms_idx.get(attachment_node)
        if fragment_attachment_atoms_idx is None:
            raise ValueError(
                f"Attachment node {attachment_node} not found in fragment graph nodes. "
                f"Available nodes: {sorted_fragment_nodes}"
            )

        if fragment_atoms_full is not None and removed_h_fragment > 0:
            # Find hydrogens bonded to attachment atom in fragment
            fragment_h_indices = [
                n for n in fragment_attachment_neighbors
                if fragment_graph.nodes[n].get('symbol') == 'H'
            ]

            # Remove the calculated number of H from fragment
            h_to_remove_from_fragment = set(fragment_h_indices[:removed_h_fragment])

            # Filter fragment atoms
            fragment_atoms = self._filter_fragment_atoms(
                fragment_graph,
                fragment_atoms_full,
                h_to_remove_from_fragment
            )
            
            # Update attachment index after filtering
            # The filtered atoms array has a different mapping
            sorted_kept_nodes = [n for n in sorted_fragment_nodes if n not in h_to_remove_from_fragment]
            fragment_attachment_atoms_idx = sorted_kept_nodes.index(attachment_node)
        else:
            fragment_atoms = fragment_atoms_full

        if fragment_atoms is not None:
            # Use geometry-aware positioning that considers bonding angles and lengths
            # Use only the kept neighbors for geometry calculation
            # fragment_attachment_atoms_idx is now the correct index in fragment_atoms array
            fragment_positions = calculate_fragment_positions_geometry_aware(
                target_atom_idx=target_orig_idx,
                target_neighbors=neighbors_to_keep,
                molecule_graph=self.graph,
                molecule_atoms=self.atoms_object,
                fragment_atoms=fragment_atoms,
                fragment_attachment_index=fragment_attachment_atoms_idx,
                fragment_graph=fragment_graph,
                parent_structure=self.parent_view.full_structure,
            )

            for i in range(len(fragment_atoms)):
                new_positions.append(fragment_positions[i])
                new_symbols.append(fragment_atoms[i].symbol)

        new_molecule = Atoms(symbols=new_symbols, positions=new_positions)

        # Reconstruct full structure
        full_structure = reconstruct_structure(
            original_structure=self.parent_view.full_structure,
            old_molecule_indices=self.original_indices,
            new_molecule_atoms=new_molecule,
        )

        return full_structure

    def _graph_to_atoms(self, graph: nx.Graph) -> Optional[Atoms]:
        """Convert a graph to Atoms object using graph converter.
        
        If coordinates are missing, generates them using RDKit 3D embedding.
        This is needed for fragments from SMILES strings that don't have coordinates.
        """
        if len(graph.nodes()) == 0:
            return None

        # Check if coordinates are present
        has_coords = all(
            graph.nodes[node].get('position') is not None
            for node in graph.nodes()
        )
        
        if not has_coords:
            # Generate coordinates using RDKit 3D embedding
            from q2D_Materials.utils.molecules.graph_converter import graph_to_rdkit, rdkit_to_graph
            
            if not RDKIT_EMBEDDING_AVAILABLE:
                raise ValueError(
                    "RDKit embedding is required to generate coordinates for fragment graphs. "
                    "Please install RDKit with AllChem support."
                )
            
            # Convert graph to RDKit Mol
            mol = graph_to_rdkit(graph, preserve_coords=False)
            
            # Sanitize molecule (calculates implicit valence, etc.)
            from rdkit import Chem
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                # If sanitization fails, try to fix it
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                except Exception:
                    pass  # Continue anyway
            
            # Generate 3D coordinates using RDKit's ETKDG method
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception:
                    # MMFF optimization failed, but embedding succeeded
                    pass
            except Exception:
                # Fallback to basic distance geometry embedding
                try:
                    rdDistGeom.EmbedMolecule(mol)
                except Exception:
                    raise ValueError(
                        "Failed to generate 3D coordinates for fragment. "
                        "RDKit embedding failed. Fragment may be too complex or invalid."
                    )
            
            # Convert back to graph with coordinates
            graph = rdkit_to_graph(mol, coords=None)
        
        # Use graph converter
        from q2D_Materials.utils.molecules.graph_converter import graph_to_atoms
        return graph_to_atoms(graph, validate_geometry=False)

    def _filter_fragment_atoms(
        self,
        fragment_graph: nx.Graph,
        fragment_atoms: Atoms,
        indices_to_remove: set
    ) -> Atoms:
        """Filter out specific atoms from fragment.

        Parameters
        ----------
        fragment_graph : nx.Graph
            Fragment molecular graph
        fragment_atoms : Atoms
            Full fragment atoms
        indices_to_remove : set of int
            Node indices in fragment_graph to remove

        Returns
        -------
        Atoms
            Filtered fragment with specified atoms removed
        """
        # Get sorted node list (same order as used in _graph_to_atoms)
        all_nodes = sorted(fragment_graph.nodes())

        # Build mapping from graph node index to atoms index
        node_to_atoms_idx = {node: i for i, node in enumerate(all_nodes)}

        # Determine which atoms indices to keep
        atoms_indices_to_keep = [
            node_to_atoms_idx[node]
            for node in all_nodes
            if node not in indices_to_remove
        ]

        # Filter symbols and positions
        symbols = fragment_atoms.get_chemical_symbols()
        positions = fragment_atoms.get_positions()

        filtered_symbols = [symbols[i] for i in atoms_indices_to_keep]
        filtered_positions = [positions[i] for i in atoms_indices_to_keep]

        return Atoms(symbols=filtered_symbols, positions=filtered_positions)

    def analyze_spacer(self, analyzer=None):
        """Analyze DJ spacer properties.

        Computes penetration depth, compression, backbone, and side chains
        for this spacer molecule.

        Parameters
        ----------
        analyzer : q2D_analyzer, optional
            Full analyzer instance for accessing halogen positions.
            If not provided, uses parent_view.analyzer if available.
            Penetration depth requires analyzer context.

        Returns
        -------
        SpacerAnalysisResult
            Complete analysis with all metrics

        Examples
        --------
        >>> from q2D_Materials.analyzer import q2D_analyzer
        >>> from q2D_Materials.modifier import GraphView
        >>>
        >>> analyzer = q2D_analyzer("structure.cif")
        >>> analyzer.analyze()
        >>> view = GraphView(analyzer)
        >>>
        >>> spacer = view.spacers[0]
        >>> analysis = spacer.analyze_spacer()
        >>> print(f"Compression: {analysis.compression_factor:.2f}")
        """
        from q2D_Materials.analyzer.characterization.spacer_analysis import SpacerAnalysis

        # Use provided analyzer or try to get from parent_view
        if analyzer is None and hasattr(self.parent_view, 'analyzer'):
            analyzer = getattr(self.parent_view, 'analyzer', None)

        spacer_analyzer = SpacerAnalysis(self, analyzer)
        return spacer_analyzer.compute()


class MoleculesList:
    """Container for molecule list with convenient access methods."""

    def __init__(self, molecules: List[MoleculeGraph]):
        self._molecules = molecules

    def list(self) -> List[MoleculeGraph]:
        """Return list of all MoleculeGraph objects."""
        return self._molecules

    def __iter__(self):
        """Allow direct iteration: for mol in view.molecules"""
        return iter(self._molecules)

    def __getitem__(self, index: int) -> MoleculeGraph:
        """Allow indexing: view.molecules[0]"""
        return self._molecules[index]

    def __len__(self) -> int:
        """Return number of molecules."""
        return len(self._molecules)
