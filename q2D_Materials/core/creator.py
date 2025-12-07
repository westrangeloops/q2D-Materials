from ase.visualize import view
from ..pipeline import create_bulk_perovskite, create_monolayer_perovskite, auto_calculate_BX_distance
from .structure import q2DStructure


class q2D_creator:
    """
    Minimal creator: only bulk perovskites using the new pipeline.
    """

    def __init__(self):
        pass

    def view_structure(self, structure):
        return view(structure)

    def create_perovskite(
        self,
        A_ions,
        B_ions,
        X_ions,
        supercell=(1, 1, 1),
        BX_dist=None,
        template="cubic",
        glazer_angles=None,
        glazer_pattern=None,
        structure_type="bulk",
        thickness=1,
        jahn_teller_dist=1.0,
        vacuum: float = 10.0,
    ):
        """
        Create a perovskite structure and wrap it in q2DStructure.
        
        Parameters
        ----------
        A_ions, B_ions, X_ions : str, list, or Atoms
            Ions for A, B, and X sites
        supercell : tuple[int, int, int]
            Supercell dimensions (nx, ny, nz). Applied using ASE's supercell mechanism.
        template : str
            Name of template; any JSON available in builders/data (e.g., 'cubic', 'reduced', or custom)
        thickness : int
            For monolayers: number of octahedra stacked in Z direction
        structure_type : str
            'bulk' or 'monolayer'
        jahn_teller_dist : float
            Optional elongation factor for c-axis in bulk cells (1.0 = none)
        vacuum : float
            Vacuum padding to add for monolayer cells (ignored for bulk)
        """
        if BX_dist is None:
            B_first = B_ions[0] if isinstance(B_ions, list) else B_ions
            X_first = X_ions[0] if isinstance(X_ions, list) else X_ions
            BX_dist = auto_calculate_BX_distance(B_first, X_first)

        if structure_type.lower() == "bulk":
            atoms = create_bulk_perovskite(
                A=A_ions,
                B=B_ions,
                X=X_ions,
                supercell=supercell,
                BX_dist=BX_dist,
                template=template,
                glazer_angles=glazer_angles,
                glazer_pattern=glazer_pattern,
                jahn_teller_dist=jahn_teller_dist,
            )
        elif structure_type.lower() == "monolayer":
            atoms = create_monolayer_perovskite(
                A=A_ions,
                B=B_ions,
                X=X_ions,
                supercell=supercell,
                thickness=thickness,
                BX_dist=BX_dist,
                template=template,
                glazer_angles=glazer_angles,
                glazer_pattern=glazer_pattern,
                jahn_teller_dist=jahn_teller_dist,
                vacuum=vacuum,
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
            supercell_size=supercell,  # Keep for backward compatibility
        )