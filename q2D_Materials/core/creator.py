from ase.visualize import view
from ..pipeline import create_bulk_perovskite, auto_calculate_BX_distance
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
        supercell_size=(1, 1, 1),
        BX_dist=None,
        template="cubic",
        glazer_angles=None,
        glazer_pattern=None,
    ):
        """
        Create a bulk perovskite (only) and wrap it in q2DStructure.
        """
        if BX_dist is None:
            B_first = B_ions[0] if isinstance(B_ions, list) else B_ions
            X_first = X_ions[0] if isinstance(X_ions, list) else X_ions
            BX_dist = auto_calculate_BX_distance(B_first, X_first)

        atoms = create_bulk_perovskite(
            A=A_ions,
            B=B_ions,
            X=X_ions,
            supercell_size=supercell_size,
            BX_dist=BX_dist,
            template=template,
            glazer_angles=glazer_angles,
            glazer_pattern=glazer_pattern,
        )

        return q2DStructure(
            atoms,
            structure_type="bulk",
            BX_dist=BX_dist,
            A_ions=A_ions,
            B_ions=B_ions,
            X_ions=X_ions,
            supercell_size=supercell_size,
        )