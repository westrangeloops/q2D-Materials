from ..pipeline import (
    create_bulk_perovskite,
    create_monolayer_perovskite,
    auto_calculate_BX_distance,
)
from .structure import q2DStructure
import numpy as np
from math import gcd
from typing import List, Tuple, Optional, Union, Dict


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
        template: str | Dict = "cubic",
        glazer_angles=None,
        glazer_pattern=None,
        structure_type="bulk",
        thickness=1,
        jahn_teller_dist=1.0,
        vacuum: float = 10.0,
        layer_sequence=None,
        passivator=None,
        penetration: float = 0.0,
        spacer=None,
        attachment_end: str | None = None,
        lattice_multipliers: Optional[List[float]] = None,
        optimizer: str = "KS",
        spacer_orientation: Optional[List[str]] = None,
        collision_strategy: str = "rotate",
    ):
        """
        Create a q2DStructure for a given inorganic template.

        Notes
        -----
        - For standard perovskites, provide A_ions, B_ions and X_ions.
        - For salt / spacer-only templates (e.g. ``template="salts"``) B_ions
          can be omitted; X_ions and spacer are typically sufficient.
        - Glazer tilting requires both B and X networks. If either B or X
          is effectively absent, Glazer parameters are ignored.
        - Lattice multipliers can be overridden per call via `lattice_multipliers`
          (default is to use values from the template JSON).
        - Layer sequence can include explicit inter-floor distances using the syntax
          `layer_sequence="L1-(1.5)-L3-M2-(3.2)-M1"`, where numbers in parentheses
          are absolute distances in Å between consecutive floors. Gaps without
          explicit distances use BX-based spacing.
        - Optimizer selects the method for placing spacers: "Off" (pure geometry),
          "KS" (Kinematic Solver, default), or "UFF" (UFF force field optimization).
        - Spacer orientation controls the plane alignment of spacers: "A" (normal to BC,
          parallel to A vector) or "B" (normal to AC, parallel to B vector). Can be a single
          value or list that cycles through orientations.
        - Collision strategy controls how atomic overlaps are resolved during spacer placement:
          "rotate" (rotate molecule around N-N axis), "nudge" (small XY translations),
          "optimize" (geometry optimization), "reject" (warn and skip), "off" (no checking).
        """
        # Determine effective BX distance:
        # - If B and X are provided, use ionic radii data.
        # - Otherwise fall back to a small default span suitable for salts.
        if BX_dist is None:
            if B_ions is not None and X_ions is not None:
                from q2D_Materials.pipeline.common import _get_first_element
                B_first = _get_first_element(B_ions)
                X_first = _get_first_element(X_ions)
                BX_dist = auto_calculate_BX_distance(B_first, X_first)
            else:
                BX_dist = 3.0

        # Normalize spacer: convert single value to list
        if spacer is not None and not isinstance(spacer, list):
            spacer = [spacer]

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
                sharp_spacer=spacer,
                lattice_multipliers=lattice_multipliers,
                optimizer=optimizer,
                spacer_orientation=spacer_orientation,
                collision_strategy=collision_strategy,
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
                passivator=passivator,
                penetration=penetration,
                sharp_spacer=spacer,
                attachment_end=attachment_end,
                lattice_multipliers=lattice_multipliers,
                optimizer=optimizer,
                spacer_orientation=spacer_orientation,
                collision_strategy=collision_strategy,
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
            sharp_spacer=spacer,
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
        
        This method takes two or more monolayer structures and stacks them with
        specified twist angles and interlayer distances. For bilayers (2 layers),
        it uses commensurate Moiré supercells. For multilayers, it finds the
        minimal common supercell.
        
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
            Total vacuum space (split evenly above/below) in Angstroms (default: 12.0).
            
        Returns
        -------
        q2DStructure
            New q2DStructure with the stacked monolayers.
            
        Raises
        ------
        ValueError
            If fewer than 2 monolayers provided, or if structure types are not 'monolayer',
            or if twist_angles/interlayer_distances lengths don't match.
        ImportError
            If pymatgen is not available.
        """
        from q2D_Materials.utils.twist_monolayer import (
            create_twisted_bilayer,
            create_twisted_multilayer
        )
        
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
        
        # Set defaults
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
        
        # For bilayer with a single twist angle, use the optimized bilayer function
        if len(monolayers) == 2 and twist_angles[0] is not None:
            m, n = twist_angles[0]
            return create_twisted_bilayer(
                monolayers[0], monolayers[1],
                m, n,
                interlayer_distance=interlayer_distances[0],
                vacuum=vacuum
            )
        
        # For multilayer or no-twist stacking, use the multilayer function
        return create_twisted_multilayer(
            monolayers,
            twist_angles,
            interlayer_distances,
            vacuum=vacuum
        )

    def Twist(self, m1, m2, m=3, n=1, interlayer_distance=11.0, vacuum=12.0):
        """
        Create a twisted bilayer from two monolayer structures.

        .. deprecated::
            Twist() is deprecated. Use twist() instead for more flexible multi-layer stacking.

        This is a simplified API for creating twisted bilayers from two monolayers.
        It is now a convenience wrapper around the more general twist() method.

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
        q2DStructure
            Twisted bilayer structure
        """
        import warnings
        warnings.warn(
            "Twist() is deprecated. Use twist() instead for more flexible multi-layer stacking.",
            DeprecationWarning,
            stacklevel=2
        )

        # Wrap the call to the new twist() method
        return self.twist(
            monolayers=[m1, m2],
            twist_angles=[(m, n)],
            interlayer_distances=[interlayer_distance],
            vacuum=vacuum
        )