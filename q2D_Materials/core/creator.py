from ..pipeline import (
    create_bulk_perovskite,
    create_monolayer_perovskite,
    auto_calculate_BX_distance,
)
from .structure import q2DStructure
from typing import List, Tuple, Optional, Dict, Union


class q2D_creator:
    """Creator for perovskite and related structures."""

    def __init__(self) -> None:
        """Initialize the q2D creator."""
        pass

    def create_structure(
        self,
        A_ions: Optional[Union[str, List[str]]] = None,
        B_ions: Optional[Union[str, List[str]]] = None,
        X_ions: Optional[Union[str, List[str]]] = None,
        xy_expansion: Tuple[int, int] = (1, 1),
        BX_dist: Optional[float] = None,
        template: Union[str, Dict] = "cubic",
        glazer_angles: Optional[List[float]] = None,
        glazer_pattern: Optional[List[str]] = None,
        structure_type: str = "bulk",
        thickness: int = 1,
        jahn_teller_dist: float = 1.0,
        vacuum: float = 10.0,
        layer_sequence: Optional[str] = None,
        passivator: Optional[Union[str, List[str]]] = None,
        penetration: float = 0.0,
        spacer: Optional[Union[str, List[str]]] = None,
        attachment_end: Optional[str] = None,
        lattice_multipliers: Optional[List[float]] = None,
        optimizer: str = "KS",
        spacer_orientation: Optional[List[str]] = None,
        collision_strategy: str = "rotate",
    ) -> q2DStructure:
        """
        Create a q2DStructure for a given inorganic template.

        Parameters
        ----------
        A_ions : str or list, optional
            A-site cation(s)
        B_ions : str or list, optional
            B-site cation(s). Can be omitted for salt/spacer-only templates.
        X_ions : str or list, optional
            X-site anion(s)
        xy_expansion : tuple, optional
            XY expansion factors (default: (1, 1))
        BX_dist : float, optional
            B-X bond distance. Auto-calculated from ionic radii if B and X provided.
        template : str or dict, optional
            Template name or template dictionary (default: "cubic")
        glazer_angles : list, optional
            Glazer tilt angles [a, b, c]
        glazer_pattern : list, optional
            Glazer pattern notation. Requires both B and X networks.
        structure_type : str, optional
            Structure type: "bulk" or "monolayer" (default: "bulk")
        thickness : int, optional
            Layer thickness (default: 1)
        jahn_teller_dist : float, optional
            Jahn-Teller distortion distance (default: 1.0)
        vacuum : float, optional
            Vacuum space for monolayers in Angstroms (default: 10.0)
        layer_sequence : str, optional
            Layer sequence string. Can include explicit distances: "L1-(1.5)-L3"
            where numbers in parentheses are absolute distances in Å.
        passivator : str or list, optional
            Passivator species for monolayers
        penetration : float, optional
            Penetration depth (default: 0.0)
        spacer : str or list, optional
            Spacer molecule(s) or atomic cation(s)
        attachment_end : str, optional
            Attachment end for spacers
        lattice_multipliers : list, optional
            Override template lattice multipliers
        optimizer : str, optional
            Spacer placement method: "Off", "KS" (default), or "UFF"
        spacer_orientation : list, optional
            Spacer plane alignment: "A" (normal to BC) or "B" (normal to AC)
        collision_strategy : str, optional
            Overlap resolution: "rotate" (default), "nudge", "optimize", "reject", "off"

        Returns
        -------
        q2DStructure
            Created structure

        Notes
        -----
        For salt/spacer-only templates (e.g., template="salts"), B_ions can be
        omitted. Glazer tilting requires both B and X networks.
        """
        if BX_dist is None:
            if B_ions is not None and X_ions is not None:
                from q2D_Materials.pipeline.common import _get_first_element

                B_first = _get_first_element(B_ions)
                X_first = _get_first_element(X_ions)
                BX_dist = auto_calculate_BX_distance(B_first, X_first)
            else:
                BX_dist = 3.0

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
            # For monolayers with thickness=1 and passivator, automatically set A_ions to None
            # so only passivator molecules are placed on surfaces (no A-sites in middle)
            effective_A_ions = A_ions
            if passivator is not None and thickness == 1:
                effective_A_ions = None
            
            atoms = create_monolayer_perovskite(
                A=effective_A_ions,
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
    ) -> q2DStructure:
        """
        Create a twisted stacking of multiple monolayer structures.

        Parameters
        ----------
        monolayers : List[q2DStructure]
            List of monolayer structures to stack (minimum 2, must have
            structure_type='monolayer')
        twist_angles : List[Tuple[int, int]], optional
            List of (m, n) tuples for twist angles, one per layer excluding first.
            If None, no rotations applied.
        interlayer_distances : List[float], optional
            Vertical distances between consecutive layers in Angstroms.
            Defaults to 11.0 Å if None.
        vacuum : float, optional
            Total vacuum space split evenly above/below in Angstroms (default: 12.0)

        Returns
        -------
        q2DStructure
            New q2DStructure with stacked monolayers

        Raises
        ------
        ValueError
            If fewer than 2 monolayers, wrong structure types, or length mismatches
        ImportError
            If pymatgen is not available
        """
        from q2D_Materials.utils.other.twist_monolayer import (
            create_twisted_bilayer,
            create_twisted_multilayer,
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
            if mono.structure_type != "monolayer":
                raise ValueError(
                    f"All structures must have structure_type='monolayer'. "
                    f"Structure {i} has structure_type='{mono.structure_type}'"
                )

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

        if len(monolayers) == 2 and twist_angles[0] is not None:
            m, n = twist_angles[0]
            return create_twisted_bilayer(
                monolayers[0],
                monolayers[1],
                m,
                n,
                interlayer_distance=interlayer_distances[0],
                vacuum=vacuum,
            )

        return create_twisted_multilayer(
            monolayers, twist_angles, interlayer_distances, vacuum=vacuum
        )

    def Twist(
        self,
        m1: q2DStructure,
        m2: q2DStructure,
        m: int = 3,
        n: int = 1,
        interlayer_distance: float = 11.0,
        vacuum: float = 12.0,
    ) -> q2DStructure:
        """
        Create a twisted bilayer from two monolayer structures.

        .. deprecated::
            Use :meth:`twist` instead for more flexible multi-layer stacking.

        Parameters
        ----------
        m1 : q2DStructure
            First monolayer structure
        m2 : q2DStructure
            Second monolayer structure
        m : int, optional
            First integer parameter for twist angle (default: 3)
        n : int, optional
            Second integer parameter for twist angle (default: 1)
        interlayer_distance : float, optional
            Vertical distance between layers in Angstroms (default: 11.0)
        vacuum : float, optional
            Vacuum space above and below in Angstroms (default: 12.0)

        Returns
        -------
        q2DStructure
            Twisted bilayer structure
        """
        import warnings

        warnings.warn(
            "Twist() is deprecated. Use twist() instead for more flexible multi-layer stacking.",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.twist(
            monolayers=[m1, m2],
            twist_angles=[(m, n)],
            interlayer_distances=[interlayer_distance],
            vacuum=vacuum,
        )