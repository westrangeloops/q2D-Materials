from ..pipeline import (
    create_bulk_perovskite,
    create_monolayer_perovskite,
    auto_calculate_BX_distance,
)
from .structure import q2DStructure
import numpy as np
from math import gcd
from typing import List, Tuple, Optional, Union


class q2D_creator:
    """Minimal creator for perovskite and related structures."""

    def __init__(self):
        pass

    def create_structure(
        self,
        A_ions=None,
        B_ions=None,
        X_ions=None,
        xy_expansion=(1, 1),
        BX_dist=None,
        template="cubic",
        glazer_angles=None,
        glazer_pattern=None,
        structure_type="bulk",
        thickness=1,
        jahn_teller_dist=1.0,
        vacuum: float = 10.0,
        layer_sequence=None,
        spacer=None,
        penetration: float = 0.0,
        sharp_spacer=None,
        attachment_end: str | None = None,
        lattice_multipliers: Optional[List[float]] = None,
        optimizer: str = "KS",
    ):
        """
        Create a q2DStructure for a given inorganic template.

        Notes
        -----
        - For standard perovskites, provide A_ions, B_ions and X_ions.
        - For salt / spacer-only templates (e.g. ``template="salts"``) B_ions
          can be omitted; X_ions and sharp_spacer are typically sufficient.
        - Glazer tilting requires both B and X networks. If either B or X
          is effectively absent, Glazer parameters are ignored.
        - Lattice multipliers can be overridden per call via `lattice_multipliers`
          (default is to use values from the template JSON).
        - Layer sequence can include explicit inter-floor distances using the syntax
          `layer_sequence="L1-(1.5)-L3-M2-(3.2)-M1"`, where numbers in parentheses
          are absolute distances in Å between consecutive floors. Gaps without
          explicit distances use BX-based spacing.
        - Optimizer selects the method for placing sharp spacers: "Off" (pure geometry),
          "KS" (Kinematic Solver, default), or "UFF" (UFF force field optimization).
        """
        # Normalize sharp_spacer: convert single value to list
        if sharp_spacer is not None and not isinstance(sharp_spacer, list):
            sharp_spacer = [sharp_spacer]

        # Determine effective BX distance:
        # - If B and X are provided, use ionic radii data.
        # - Otherwise fall back to a small default span suitable for salts.
        if BX_dist is None:
            if B_ions is not None and X_ions is not None:
                B_first = B_ions[0] if isinstance(B_ions, list) else B_ions
                X_first = X_ions[0] if isinstance(X_ions, list) else X_ions
                BX_dist = auto_calculate_BX_distance(B_first, X_first)
            else:
                BX_dist = 3.0

        # Normalize sharp_spacer: convert single value to list
        if sharp_spacer is not None and not isinstance(sharp_spacer, list):
            sharp_spacer = [sharp_spacer]

        if structure_type.lower() == "bulk":
            atoms = create_bulk_perovskite(
                A=A_ions,
                B=B_ions,
                X=X_ions,
                xy_expansion=xy_expansion,
                BX_dist=BX_dist,
                template=template,
                glazer_angles=glazer_angles,
                glazer_pattern=glazer_pattern,
                jahn_teller_dist=jahn_teller_dist,
                thickness=thickness,
                layer_sequence=layer_sequence,
                sharp_spacer=sharp_spacer,
                lattice_multipliers=lattice_multipliers,
                optimizer=optimizer,
            )
        elif structure_type.lower() == "monolayer":
            atoms = create_monolayer_perovskite(
                A=A_ions,
                B=B_ions,
                X=X_ions,
                xy_expansion=xy_expansion,
                thickness=thickness,
                BX_dist=BX_dist,
                template=template,
                glazer_angles=glazer_angles,
                glazer_pattern=glazer_pattern,
                jahn_teller_dist=jahn_teller_dist,
                vacuum=vacuum,
                layer_sequence=layer_sequence,
                spacer=spacer,
                penetration=penetration,
                sharp_spacer=sharp_spacer,
                attachment_end=attachment_end,
                lattice_multipliers=lattice_multipliers,
                optimizer=optimizer,
            )
        else:
            raise ValueError(f"structure_type must be 'bulk' or 'monolayer', got '{structure_type}'")

        return q2DStructure(
            atoms,
            structure_type=structure_type,
            BX_dist=BX_dist,
            A_ions=A_ions,
            B_ions=B_ions,
            X_ions=X_ions,
            xy_expansion=xy_expansion,
            sharp_spacer=sharp_spacer,
        )


    def twist(
        self,
        monolayers: List[q2DStructure],
        twist_angles: Optional[List[Tuple[int, int]]] = None,
        interlayer_distances: Optional[List[float]] = None,
        vacuum: float = 12.0,
    ):
        """
        Create a twisted stacking of multiple monolayer structures.
        
        This method takes two or more monolayer structures, finds the minimal common
        supercell (LCM of xy_expansions), and returns a stacked structure with optional
        rotations applied to each layer.
        
        Parameters
        ----------
        monolayers : List[q2DStructure]
            List of monolayer structures to stack (must have structure_type='monolayer').
            Minimum 2 structures required.
        twist_angles : List[Tuple[int, int]], optional
            List of (m, n) tuples for twist angles. One per layer (excluding first).
            If None, no rotations are applied. Length should be len(monolayers) - 1.
        interlayer_distances : List[float], optional
            Vertical distances between consecutive layers in Angstroms.
            If None, defaults to 11.0 Å for all interlayer gaps.
            Length should be len(monolayers) - 1.
        vacuum : float, optional
            Vacuum space outside the stacked structure in Angstroms (default: 12.0).
            This adds vacuum above the top layer and below the bottom layer.
            
        Returns
        -------
        q2DStructure
            New q2DStructure with the stacked monolayers in a common supercell.
            
        Raises
        ------
        ValueError
            If fewer than 2 monolayers provided, or if structure types are not 'monolayer',
            or if twist_angles/interlayer_distances lengths don't match.
        ImportError
            If pymatgen is not available.
        """
        if len(monolayers) < 2:
            raise ValueError(
                f"twist() requires at least 2 monolayer structures, got {len(monolayers)}"
            )
        
        for i, mono in enumerate(monolayers):
            if not isinstance(mono, q2DStructure):
                raise TypeError(
                    f"All structures must be q2DStructure instances. "
                    f"Structure {i} is {type(mono)}"
                )
            if mono.structure_type != 'monolayer':
                raise ValueError(
                    f"All structures must have structure_type='monolayer'. "
                    f"Structure {i} has structure_type='{mono.structure_type}'"
                )
        
        try:
            from pymatgen.core import Structure, Lattice
            from pymatgen.io.ase import AseAtomsAdaptor
        except ImportError:
            raise ImportError(
                "pymatgen is required for twist() method. Please install pymatgen."
            )
        
        adapter = AseAtomsAdaptor()
        
        def lcm(a: int, b: int) -> int:
            """Calculate Least Common Multiple of two integers."""
            return abs(a * b) // gcd(a, b) if a and b else 0
        
        def lcm_list(numbers: List[int]) -> int:
            """Calculate LCM of a list of integers."""
            result = numbers[0]
            for num in numbers[1:]:
                result = lcm(result, num)
            return result
        
        xy_expansions = []
        for mono in monolayers:
            exp = mono.xy_expansion if mono.xy_expansion else (1, 1)
            xy_expansions.append(exp)
        
        nx_values = [exp[0] for exp in xy_expansions]
        ny_values = [exp[1] for exp in xy_expansions]
        
        common_nx = lcm_list(nx_values)
        common_ny = lcm_list(ny_values)
        
        expanded_structures = []
        for i, mono in enumerate(monolayers):
            mono_pmg = adapter.get_structure(mono)
            
            exp = xy_expansions[i]
            scale_x = common_nx // exp[0]
            scale_y = common_ny // exp[1]
            
            if scale_x > 1 or scale_y > 1:
                mono_pmg.make_supercell([[scale_x, 0, 0],
                                        [0, scale_y, 0],
                                        [0, 0, 1]])
            
            expanded_structures.append(mono_pmg)
        
        if twist_angles is None:
            twist_angles = [None] * (len(monolayers) - 1)
        elif len(twist_angles) != len(monolayers) - 1:
            raise ValueError(
                f"twist_angles must have length {len(monolayers) - 1} "
                f"(one per layer except first), got {len(twist_angles)}"
            )
        
        if interlayer_distances is None:
            interlayer_distances = [11.0] * (len(monolayers) - 1)
        elif len(interlayer_distances) != len(monolayers) - 1:
            raise ValueError(
                f"interlayer_distances must have length {len(monolayers) - 1} "
                f"(one per interlayer gap), got {len(interlayer_distances)}"
            )
        
        all_coords = []
        all_species = []
        z_offset = 0.0
        
        for i, (mono_pmg, mono) in enumerate(zip(expanded_structures, monolayers)):
            coords = np.array([site.coords for site in mono_pmg], dtype=np.float64)
            
            if i > 0 and twist_angles[i - 1] is not None:
                m, n = twist_angles[i - 1]
                cost = (m**2 - n**2) / (m**2 + n**2)
                sint = (2 * m * n) / (m**2 + n**2)
                rot_matrix = np.array([[cost, -sint, 0],
                                       [sint, cost, 0],
                                       [0, 0, 1]], dtype=np.float64)
                coords = coords @ rot_matrix.T
            
            coords[:, 2] += z_offset
            all_coords.append(coords)
            all_species.extend([site.species for site in mono_pmg])
            
            if i < len(monolayers) - 1:
                z_offset -= interlayer_distances[i]
        
        all_coords = np.vstack(all_coords)
        
        base_lattice = expanded_structures[0].lattice
        stacked = Structure(base_lattice, all_species, all_coords, coords_are_cartesian=True)
        
        stacked_atoms = adapter.get_atoms(stacked)
        
        positions = stacked_atoms.positions
        min_z = np.min(positions[:, 2])
        max_z = np.max(positions[:, 2])
        
        z_shift = vacuum / 2.0 - min_z
        positions[:, 2] += z_shift
        
        new_z_length = max_z - min_z + vacuum
        
        current_cell = stacked_atoms.cell
        stacked_atoms.cell = [
            current_cell[0],
            current_cell[1],
            [0, 0, new_z_length]
        ]
        
        combined_metadata = {}
        if hasattr(monolayers[0], '_metadata') and monolayers[0]._metadata:
            combined_metadata = monolayers[0]._metadata.copy()
        combined_metadata['twist_params'] = twist_angles
        combined_metadata['interlayer_distances'] = interlayer_distances
        combined_metadata['vacuum'] = vacuum
        combined_metadata['common_supercell'] = (common_nx, common_ny)
        combined_metadata['n_layers'] = len(monolayers)
        
        return q2DStructure(
            stacked_atoms,
            structure_type='monolayer',
            BX_dist=monolayers[0].BX_dist,
            A_ions=monolayers[0].A_ions,
            B_ions=monolayers[0].B_ions,
            X_ions=monolayers[0].X_ions,
            xy_expansion=(common_nx, common_ny),
            **combined_metadata
        )

    def Twist(self, m1, m2, m=3, n=1, interlayer_distance=11.0, vacuum=12.0):
        """
        Create a twisted bilayer from two monolayer structures.

        This is a simplified API for creating twisted bilayers from two monolayers.
        The heavy lifting is done by create_twisted_bilayer in twist_monolayer.py.

        Parameters
        ----------
        m1 : q2DStructure
            First monolayer structure
        m2 : q2DStructure
            Second monolayer structure
        m : int, optional
            First integer parameter for twist angle calculation (default: 3)
        n : int, optional
            Second integer parameter for twist angle calculation (default: 1)
        interlayer_distance : float, optional
            Vertical distance between layers in Angstroms (default: 11.0)
        vacuum : float, optional
            Vacuum space above and below the bilayer in Angstroms (default: 12.0)

        Returns
        -------
        Atoms
            ASE Atoms object of the twisted bilayer
        """
        from q2D_Materials.utils.twist_monolayer import create_twisted_bilayer
        return create_twisted_bilayer(m1, m2, m, n, interlayer_distance, vacuum)