from ..pipeline import create_bulk_perovskite, create_monolayer_perovskite, auto_calculate_BX_distance
from .structure import q2DStructure


class q2D_creator:
    """Minimal creator for perovskite structures."""

    def __init__(self):
        pass

    def create_perovskite(
        self,
        A_ions,
        B_ions,
        X_ions,
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
    ):
        if BX_dist is None:
            B_first = B_ions[0] if isinstance(B_ions, list) else B_ions
            X_first = X_ions[0] if isinstance(X_ions, list) else X_ions
            BX_dist = auto_calculate_BX_distance(B_first, X_first)

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
                layer_sequence=layer_sequence,
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
        )