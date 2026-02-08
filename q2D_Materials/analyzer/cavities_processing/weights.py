"""Weight configuration for cavity X atom selection.

Used by A-site (cuboctahedron) and spacer (antiprism) cavity detection
to score and select which X atoms form the cage edges.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ASiteWeights:
    """Weight configuration for A-site cavity X atom selection.

    Scoring formula: score = w_dist_anchor * dist_anchor + w_z_diff * z_diff
    Lower scores are preferred (atoms closer to anchor and at similar Z).

    Parameters
    ----------
    w_dist_anchor : float
        Weight for distance to anchor (A-site geometric center). Default: 0.7
    w_z_diff : float
        Weight for Z-coordinate difference from anchor. Default: 0.3
    """
    w_dist_anchor: float = 0.7
    w_z_diff: float = 0.3

    def __post_init__(self):
        """Validate weights sum to 1.0 for interpretability."""
        total = self.w_dist_anchor + self.w_z_diff
        if not np.isclose(total, 1.0, atol=1e-6):
            import warnings
            warnings.warn(
                f"A-site weights sum to {total:.3f}, not 1.0. "
                "Weights should ideally sum to 1.0 for interpretability.",
                UserWarning
            )


@dataclass
class SpacerWeights:
    """Weight configuration for spacer cavity terminal X atom selection.

    Scoring formula: score = w_dist_anchor * dist_anchor +
                             w_dist_b_center * dist_b_center +
                             w_z_diff * z_diff
    Lower scores are preferred.

    Parameters
    ----------
    w_dist_anchor : float
        Weight for distance to anchor (NH3 N atom). Default: 0.4
    w_dist_b_center : float
        Weight for distance to B atom geometric center. Default: 0.3
    w_z_diff : float
        Weight for Z-coordinate difference from anchor. Default: 0.3
    """
    w_dist_anchor: float = 0.4
    w_dist_b_center: float = 0.3
    w_z_diff: float = 0.3

    def __post_init__(self):
        """Validate weights sum to 1.0 for interpretability."""
        total = self.w_dist_anchor + self.w_dist_b_center + self.w_z_diff
        if not np.isclose(total, 1.0, atol=1e-6):
            import warnings
            warnings.warn(
                f"Spacer weights sum to {total:.3f}, not 1.0. "
                "Weights should ideally sum to 1.0 for interpretability.",
                UserWarning
            )
