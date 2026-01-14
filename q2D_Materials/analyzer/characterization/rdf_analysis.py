"""
Radial distribution function (RDF) analysis for perovskite structures.

This module provides functions to compute partial radial distribution functions
for element pairs in perovskite structures.

Functions
---------
_calculate_partial_rdf
    Compute partial radial distribution functions for element pairs
_gaussian_kernel_discrete_spectrum
    Smooth discrete spectrum with gaussian kernel for RDF
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


def _gaussian_kernel_discrete_spectrum(
    spectrum: np.ndarray,
    smearing: Optional[float] = None,
    gridpoints: int = 200,
    v_min: Optional[float] = None,
    v_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth the input spectrum with a gaussian kernel over the supported region.

    Parameters
    ----------
    spectrum : np.ndarray
        Array of distance values to smooth
    smearing : float, optional
        Gaussian smearing width. If None, uses (max - min) / 50
    gridpoints : int
        Number of grid points for output (default: 200)
    v_min : float, optional
        Minimum value for grid. If None, uses min(spectrum) - 3*smearing
    v_max : float, optional
        Maximum value for grid. If None, uses max(spectrum) + 3*smearing

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (xs, smoothed_spectrum) - grid points and smoothed values
    """
    spectrum = np.asarray(spectrum, dtype=float)

    if smearing is None:
        if len(spectrum) > 0:
            smearing = (np.max(spectrum) - np.min(spectrum)) / 50
        else:
            smearing = 0.1

    if v_min is None:
        if len(spectrum) > 0:
            v_min = np.min(spectrum) - 3 * smearing
        else:
            v_min = 0.0

    if v_max is None:
        if len(spectrum) > 0:
            v_max = np.max(spectrum) + 3 * smearing
        else:
            v_max = 10.0

    xs = np.linspace(v_min, v_max, gridpoints)
    res = np.zeros_like(xs)

    for delta in spectrum:
        res += np.exp(-((xs - delta) ** 2) / (smearing ** 2))

    return xs, res


def _calculate_partial_rdf(
    analyzer,
    element_pairs: List[List[str]],
    max_dist: float = 10.0,
    npoints: Optional[int] = None,
    ss_norm: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute partial radial distribution functions for element pairs.

    Uses gaussian kernel smoothing for smooth RDFs suitable for perovskite structures.

    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzer instance with analyzed structure
    element_pairs : List[List[str]]
        List of [element1, element2] pairs to compute RDFs for
    max_dist : float
        Maximum distance in Angstroms for RDF computation
    npoints : int, optional
        Number of grid points. If None, uses max_dist * 50
    ss_norm : bool
        Whether to apply structure-specific normalization

    Returns
    -------
    dict
        Dictionary with keys like "PbI_x", "PbI_rdf" for each pair
    """
    atoms = analyzer.cell

    # Set up supercell if needed
    cell = atoms.cell
    cell_norms = [np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), np.linalg.norm(cell[2])]
    supercell = [0, 0, 0]
    for idx, i in enumerate(cell_norms):
        scl = int(max_dist * 2 / i) + 1
        supercell[idx] = scl
    tmp_atoms = atoms.repeat(tuple(supercell))

    plottable_rdfs = {}

    if npoints is None:
        npoints = int(max_dist * 50)

    # Get all distances
    dm = tmp_atoms.get_all_distances(mic=True)
    curr_max = 0

    for pair in element_pairs:
        pair_dists = []
        i_indices = np.where(tmp_atoms.symbols == pair[0])[0]
        phi = len(i_indices)

        for i in i_indices:
            for j in np.where(tmp_atoms.symbols == pair[1])[0]:
                if i != j:
                    pair_dists.append(dm[i, j])

        # Smooth with gaussian kernel
        this_xs, this_spectra = _gaussian_kernel_discrete_spectrum(
            pair_dists,
            smearing=0.05,
            gridpoints=npoints,
            v_min=0,
            v_max=max_dist,
        )

        if ss_norm:
            this_spectra *= phi

        pair_key = "".join(pair)
        plottable_rdfs[pair_key + "_x"] = this_xs
        plottable_rdfs[pair_key + "_rdf"] = this_spectra

        if curr_max < np.max(this_spectra):
            curr_max = np.max(this_spectra)

    # Normalize all RDFs by maximum
    for pair in element_pairs:
        pair_key = "".join(pair)
        if pair_key + "_rdf" in plottable_rdfs:
            plottable_rdfs[pair_key + "_rdf"] /= curr_max

    return plottable_rdfs

