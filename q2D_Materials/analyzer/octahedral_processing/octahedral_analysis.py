"""Octahedral analysis functions for perovskite structures.

Parts of this code are from PDynA (https://github.com/WMD-group/PDynA):
MIT License - Copyright (c) 2022 Xia Liang
"""

from typing import Optional, Tuple, List, Dict, Any, Union
import logging
import numpy as np
from scipy.spatial.transform import Rotation as sstr
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.molecule_matcher import HungarianOrderMatcher

from ...utils.geometry.structural_constants import (
    DEFAULT_FITTING_TOLERANCE,
    DEFAULT_RMSD_THRESHOLD,
    DEFAULT_ANGLE_TOLERANCE,
)
from ...utils.geometry.structural_utils import (
    apply_pbc_cart_vecs,
    get_frac_from_cart,
    get_cart_from_frac,
    distance_matrix_handler,
    periodicity_fold,
)

logger = logging.getLogger(__name__)


def load_octahedral_basis(filename_basis: str) -> Tuple[Dict[str, Any], int]:
    """Load octahedral basis dictionary from JSON file.
    
    Parameters
    ----------
    filename_basis : str
        Path to the basis JSON file
    
    Returns
    -------
    tuple of (dict, int)
        Basis dictionary and distortion dimension
    """
    import json
    import os
    
    try:
        with open(filename_basis, 'r') as f:
            dict_basis = json.load(f)
            dist_dim = len(dict_basis) - 3
            return dict_basis, dist_dim
    except IOError as e:
        raise FileNotFoundError(f"Could not read basis file: {filename_basis}") from e


def octahedra_coords_into_bond_vectors(
    raw: np.ndarray,
    mymat: np.ndarray,
) -> np.ndarray:
    """Convert octahedron coordinates to six B-X bond vectors.
    
    Parameters
    ----------
    raw : np.ndarray
        Raw coordinates of an octahedron, shape (6, 3)
    mymat : np.ndarray
        Lattice matrix, shape (3, 3)
    
    Returns
    -------
    np.ndarray
        Six B-X bond vectors, shape (6, 3)
    """
    bx1 = apply_pbc_cart_vecs(raw, mymat)
    bx = bx1 / np.mean(np.linalg.norm(bx1, axis=1))
    return bx


def calc_distortions_from_bond_vectors(
    bx: np.ndarray,
    dict_basis: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute tilting and distortion from six B-X bond vectors.
    
    Parameters
    ----------
    bx : np.ndarray
        Six B-X bond vectors, shape (6, 3)
    dict_basis : dict
        Basis dictionary for distortion calculation
    
    Returns
    -------
    tuple of (np.ndarray, np.ndarray, float)
        Distortion amplitudes, rotation matrix, RMSD
    """
    ideal_coords = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                    [0, 0, 1], [0, 1, 0], [1, 0, 0]]

    irrep_distortions = []
    for irrep in dict_basis.keys():
        for elem in dict_basis[irrep]:
            irrep_distortions.append(elem)

    pymatgen_molecule = Molecule(
        species=[Element("Pb"), Element("H"), Element("He"), Element("Li"),
                 Element("Be"), Element("B"), Element("I")],
        coords=np.concatenate((np.zeros((1, 3)), bx), axis=0))

    pymatgen_molecule_ideal = Molecule(
        species=pymatgen_molecule.species,
        coords=np.concatenate((np.zeros((1, 3)), ideal_coords), axis=0))

    pymatgen_molecule, rotmat, rmsd = match_molecules_extra(
        pymatgen_molecule, pymatgen_molecule_ideal)

    distortion_amplitudes = calc_displacement(
        pymatgen_molecule, pymatgen_molecule_ideal, irrep_distortions
    )

    distortion_amplitudes = distortion_amplitudes * distortion_amplitudes
    temp_list = []
    count = 0
    for irrep in dict_basis:
        dim = len(dict_basis[irrep])
        temp_list.append(np.sum(distortion_amplitudes[count:count + dim]))
        count += dim
    distortion_amplitudes = np.sqrt(temp_list)[3:]

    return distortion_amplitudes, rotmat, rmsd


def calc_distortions_from_bond_vectors_full(
    bx: np.ndarray,
    dict_basis: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute tilting and distortion from six B-X bond vectors (full version).
    
    Parameters
    ----------
    bx : np.ndarray
        Six B-X bond vectors, shape (6, 3)
    dict_basis : dict
        Basis dictionary for distortion calculation
    
    Returns
    -------
    tuple of (np.ndarray, np.ndarray, float)
        Distortion amplitudes, rotation matrix, RMSD
    """
    ideal_coords = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                    [0, 0, 1], [0, 1, 0], [1, 0, 0]]

    irrep_distortions = []
    for irrep in dict_basis.keys():
        for elem in dict_basis[irrep]:
            irrep_distortions.append(elem)

    pymatgen_molecule = Molecule(
        species=[Element("Pb"), Element("H"), Element("He"), Element("Li"),
                 Element("Be"), Element("B"), Element("I")],
        coords=np.concatenate((np.zeros((1, 3)), bx), axis=0))

    pymatgen_molecule_ideal = Molecule(
        species=pymatgen_molecule.species,
        coords=np.concatenate((np.zeros((1, 3)), ideal_coords), axis=0))

    pymatgen_molecule, rotmat, rmsd = match_molecules_extra(
        pymatgen_molecule, pymatgen_molecule_ideal)

    distortion_amplitudes = calc_displacement_full(
        pymatgen_molecule, pymatgen_molecule_ideal, irrep_distortions
    )

    distortion_amplitudes = distortion_amplitudes * distortion_amplitudes
    temp_list = []
    count = 0
    for irrep in dict_basis:
        dim = len(dict_basis[irrep])
        temp_list.append(np.sum(distortion_amplitudes[count:count + dim]))
        count += dim
    distortion_amplitudes = np.sqrt(temp_list)[3:]

    return distortion_amplitudes, rotmat, rmsd


def match_molecules_extra(
    molecule_transform: Molecule,
    molecule_reference: Molecule,
) -> Tuple[Molecule, np.ndarray, float]:
    """Hungarian matching to output the transformation matrix.
    
    Parameters
    ----------
    molecule_transform : pymatgen.Molecule
        The molecule to be transformed
    molecule_reference : pymatgen.Molecule
        The reference molecule
    
    Returns
    -------
    tuple of (pymatgen.Molecule, np.ndarray, float)
        Transformed molecule, rotation matrix, RMSD
    """
    (inds, u, v, rmsd) = HungarianOrderMatcher(
        molecule_reference).match(molecule_transform)

    molecule_transform.apply_operation(SymmOp(
        np.concatenate((
            np.concatenate((u.T, v.reshape(3, 1)), axis=1),
            [np.zeros(4)]), axis=0
        )))
    molecule_transform._sites = np.array(
        molecule_transform._sites)[inds].tolist()

    return molecule_transform, u, rmsd


def calc_displacement(
    pymatgen_molecule: Molecule,
    pymatgen_molecule_ideal: Molecule,
    irrep_distortions: List[np.ndarray],
) -> np.ndarray:
    """Compute displacement of atoms in the molecule for matching.
    
    Parameters
    ----------
    pymatgen_molecule : pymatgen.Molecule
        The molecule to be transformed
    pymatgen_molecule_ideal : pymatgen.Molecule
        The reference molecule
    irrep_distortions : list of np.ndarray
        Basis irreps for matching
    
    Returns
    -------
    np.ndarray
        Processed displacement of the atoms
    """
    return np.tensordot(irrep_distortions,
                        (pymatgen_molecule.cart_coords - 
                         pymatgen_molecule_ideal.cart_coords).ravel()[3:],
                        axes=1)


def calc_displacement_full(
    pymatgen_molecule: Molecule,
    pymatgen_molecule_ideal: Molecule,
    irrep_distortions: List[np.ndarray],
) -> np.ndarray:
    """Compute displacement of atoms in the molecule for matching (full version).
    
    Parameters
    ----------
    pymatgen_molecule : pymatgen.Molecule
        The molecule to be transformed
    pymatgen_molecule_ideal : pymatgen.Molecule
        The reference molecule
    irrep_distortions : list of np.ndarray
        Basis irreps for matching
    
    Returns
    -------
    np.ndarray
        Processed displacement of the atoms
    """
    return np.tensordot(irrep_distortions,
                        (pymatgen_molecule.cart_coords - 
                         pymatgen_molecule_ideal.cart_coords).ravel(),
                        axes=1)


def match_bx_orthogonal(
    bx: np.ndarray,
    fitting_tol: float = DEFAULT_FITTING_TOLERANCE,
) -> List[int]:
    """Find order of atoms in octahedron through matching with reference.
    
    Used in structure_type 1.
    
    Parameters
    ----------
    bx : np.ndarray
        Six B-X bond vectors, shape (6, 3)
    fitting_tol : float
        Fitting tolerance threshold
    
    Returns
    -------
    list of int
        Order of atoms matching to the reference
    """
    ideal_coords = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                             [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    order = []
    for ix in range(6):
        fits = np.dot(bx, ideal_coords[ix, :])
        if not (fits[fits.argsort()[-1]] - fits[fits.argsort()[-2]] > fitting_tol):
            raise ValueError(
                f"Fitting of octahedron config to ideal coords failed. "
                f"Confidence: {round(fits[fits.argsort()[-1]] - fits[fits.argsort()[-2]], 3)}, "
                f"tolerance: {fitting_tol}. Structure may be too tilted or distorted.")
        order.append(np.argmax(fits))
    if len(set(order)) != 6:
        raise ValueError("Duplicate indices in order list")
    return order


def match_bx_orthogonal_rotated(
    bx: np.ndarray,
    rotmat: np.ndarray,
    fitting_tol: float = DEFAULT_FITTING_TOLERANCE,
) -> List[int]:
    """Find order of atoms in octahedron through matching with rotated reference.
    
    Used in structure_type 2.
    
    Parameters
    ----------
    bx : np.ndarray
        Six B-X bond vectors, shape (6, 3)
    rotmat : np.ndarray
        Rotation matrix for alignment, shape (3, 3)
    fitting_tol : float
        Fitting tolerance threshold
    
    Returns
    -------
    list of int
        Order of atoms matching to the reference
    """
    ideal_coords = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                             [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    ideal_coords = np.matmul(ideal_coords, rotmat)
    order = []
    for ix in range(6):
        fits = np.dot(bx, ideal_coords[ix, :])
        if not (fits[fits.argsort()[-1]] - fits[fits.argsort()[-2]]) > fitting_tol:
            logger.warning(f"Low fitting confidence: {bx}, {fits}")
            raise ValueError(
                f"Fitting of octahedron config to ideal coords failed. "
                f"Confidence: {round(fits[fits.argsort()[-1]] - fits[fits.argsort()[-2]], 3)}, "
                f"tolerance: {fitting_tol}. Structure may be too tilted or distorted.")
        order.append(np.argmax(fits))
    assert len(set(order)) == 6
    return order


def quick_match_octahedron(
    bx: np.ndarray,
    dict_basis: Optional[Dict[str, Any]] = None,
    fitting_tol: float = DEFAULT_FITTING_TOLERANCE,
    rmsd_threshold: float = DEFAULT_RMSD_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute rotation status of an octahedron with a reference.
    
    Used in connectivity type 3.
    
    Parameters
    ----------
    bx : np.ndarray
        Six B-X bond vectors, shape (6, 3)
    dict_basis : dict
        Basis dictionary
    fitting_tol : float
        Fitting tolerance threshold
    rmsd_threshold : float
        RMSD threshold for matching
    
    Returns
    -------
    tuple of (np.ndarray, np.ndarray, float)
        Inverse rotation matrix, rotated coordinates, RMSD
    """
    def calc_match(bx_local):
        pymatgen_molecule = Molecule(
            species=[Element("Pb"), Element("I"), Element("I"), Element("I"),
                     Element("I"), Element("I"), Element("I")],
            coords=np.concatenate((np.zeros((1, 3)), bx_local), axis=0))

        pymatgen_molecule_ideal = Molecule(
            species=pymatgen_molecule.species,
            coords=np.concatenate((np.zeros((1, 3)), ideal_coords), axis=0))

        new_molecule, rotmat, rmsd = match_molecules_extra(
            pymatgen_molecule_ideal, pymatgen_molecule)
        
        return new_molecule, rotmat, rmsd
        
    ideal_coords = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                    [0, 0, 1], [0, 1, 0], [1, 0, 0]]

    new_molecule, rotmat, rmsd = calc_match(bx)
    
    if rmsd > rmsd_threshold:
        order = []
        for ix in range(6):
            fits = np.dot(bx, np.array(ideal_coords)[ix, :])
            if not (fits[fits.argsort()[-1]] - fits[fits.argsort()[-2]] > fitting_tol):
                raise ValueError(
                    f"Fitting of initial octahedron config to ideal coords failed. "
                    f"Confidence: {round(fits[fits.argsort()[-1]] - fits[fits.argsort()[-2]], 3)}, "
                    f"tolerance: {fitting_tol}. Structure may be too tilted or distorted.")
            order.append(np.argmax(fits))
        new_molecule, rotmat, rmsd = calc_match(bx[order, :])
        if rmsd > rmsd_threshold:
            logger.warning(f"High RMSD after reordering: {bx[order, :]}")
        
    new_coords = np.matmul(ideal_coords, rotmat)
    
    return np.linalg.inv(rotmat), new_coords, rmsd


def match_bx_arbitrary(
    bx: np.ndarray,
    dict_basis: Dict[str, Any],
    fitting_tol: float = 0.4,
) -> Tuple[List[int], np.ndarray, float]:
    """Find order of atoms in octahedron through matching with reference.
    
    Used in structure_type 3.
    
    Parameters
    ----------
    bx : np.ndarray
        Six B-X bond vectors, shape (6, 3)
    dict_basis : dict
        Basis dictionary
    fitting_tol : float
        Fitting tolerance threshold
    
    Returns
    -------
    tuple of (list of int, np.ndarray, float)
        Order of atoms, rotation matrices, RMSD
    """
    if dict_basis is None:
        dict_basis = {}
    ref_rot, ideal_coords, rmsd = quick_match_octahedron(bx, dict_basis, fitting_tol)
    
    order = []
    for ix in range(6):
        fits = np.dot(bx, ideal_coords[ix, :])
        if not (fits[fits.argsort()[-1]] - fits[fits.argsort()[-2]]) > fitting_tol:
            logger.warning(f"Low fitting confidence: {bx}, {fits}")
            raise ValueError(
                "Fitting of initial octahedron config to rotated reference failed. "
                "Structure may be too distorted.")
        order.append(np.argmax(fits))
    assert len(set(order)) == 6
    return order, ref_rot, rmsd


def calc_rotation_from_arbitrary_order(
    bx: np.ndarray,
    rmsd_threshold: float = DEFAULT_RMSD_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rotation of six bond vectors with arbitrary order.
    
    Used in non-ortho structure.
    
    Parameters
    ----------
    bx : np.ndarray
        Six B-X bond vectors, shape (6, 3)
    rmsd_threshold : float
        RMSD threshold for validation
    
    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Rotation angles, rotation matrix
    """
    ideal_coords = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                    [0, 0, 1], [0, 1, 0], [1, 0, 0]]

    pymatgen_molecule = Molecule(
        species=[Element("Pb"), Element("H"), Element("H"), Element("H"),
                 Element("H"), Element("H"), Element("H")],
        coords=np.concatenate((np.zeros((1, 3)), bx), axis=0))

    pymatgen_molecule_ideal = Molecule(
        species=pymatgen_molecule.species,
        coords=np.concatenate((np.zeros((1, 3)), ideal_coords), axis=0))

    pymatgen_molecule, rotmat, rmsd = match_molecules_extra(
        pymatgen_molecule, pymatgen_molecule_ideal)
    
    if rmsd > rmsd_threshold:
        raise ValueError(
            f"RMSD of fitting related to non-orthogonal frame mapping is too large "
            f"(RMSD = {round(rmsd, 4)})")
    
    rots = sstr.from_matrix(rotmat).as_euler('xyz', degrees=True)
    rots = periodicity_fold(rots)
    
    for i in range(len(rots)):
        if abs(rots[i]) < 0.01:
            rots[i] = 0
            
    rotmat = sstr.from_rotvec(rots / 180 * np.pi).as_matrix()
            
    return rots, rotmat


def find_population_gap(
    r: np.ndarray,
    find_range: List[float],
    init: np.ndarray,
    tol: int = 0,
) -> float:
    """Find 1D classifier value to separate two populations.
    
    Parameters
    ----------
    r : np.ndarray
        Input distance matrix array
    find_range : list of float
        Distance range to find the classifier
    init : np.ndarray
        Initial guess of the classifier
    tol : int
        Tolerance of the classifier
    
    Returns
    -------
    float
        Classifier value
    """
    from scipy.cluster.vq import kmeans
    import matplotlib.pyplot as plt
    
    scan = r.reshape(-1,)
    scan = scan[np.logical_and(scan < find_range[1], scan > find_range[0])]
    centers = kmeans(scan, k_or_guess=init, iter=20, thresh=1e-05)[0]
    p = np.mean(centers)

    y, bin_edges = np.histogram(scan, bins=50)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    if y[(np.abs(bin_centers - p)).argmin()] > tol:
        p1 = centers[0] + (centers[1] - centers[0]) / (centers[1] + centers[0]) * centers[0]
        p = p1
        if y[(np.abs(bin_centers - p)).argmin()] > tol:
            plt.hist(r.reshape(-1,), bins=100, range=[1, 10])
            raise ValueError(
                "Can't separate the different neighbours. Check fpg_val values or "
                "if initial structure is defected or too distorted.")

    return p


def convert_xb_to_bx(xb: np.ndarray) -> np.ndarray:
    """Convert X-to-B connectivity matrix to B-to-X connectivity matrix.
    
    Parameters
    ----------
    xb : np.ndarray
        X-to-B connectivity matrix
    
    Returns
    -------
    np.ndarray
        B-X connectivity matrix, shape (N, 6)
    """
    bx = [[] for i in range(int(xb.shape[1] / 3))]
    for i in range(xb.shape[1]):
        bx[xb[0, i]].append(i)
        bx[xb[1, i]].append(i)
    bx = np.array(bx)
    assert bx.shape == (int(xb.shape[1] / 3), 6)
    
    return bx


def fit_octahedral_network_frame(
    bpos_frame: np.ndarray,
    xpos_frame: np.ndarray,
    r: np.ndarray,
    mymat: np.ndarray,
    fpg_val_bx: List[float],
    rotated: bool,
    rotmat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Resolve the octahedral connectivity.
    
    Parameters
    ----------
    bpos_frame : np.ndarray
        Coordinate array of B-sites, shape (N_B, 3)
    xpos_frame : np.ndarray
        Coordinate array of X-sites, shape (N_X, 3)
    r : np.ndarray
        Distance matrix of B-X sites
    mymat : np.ndarray
        Lattice matrix, shape (3, 3)
    fpg_val_bx : list of float
        Defined B-X bond information
    rotated : bool
        If the structure is rotated
    rotmat : np.ndarray, optional
        Rotation matrix of the structure
    
    Returns
    -------
    np.ndarray
        B-X connectivity matrix, shape (N_B, 6)
    """
    max_bx_distance = find_population_gap(r, fpg_val_bx[0], fpg_val_bx[1])
        
    neigh_list = np.zeros((bpos_frame.shape[0], 6))
    for b_site, x_list in enumerate(r):
        x_idx = [i for i in range(len(x_list)) if x_list[i] < max_bx_distance]
        if len(x_idx) != 6:
            raise ValueError(
                f"Number of X site atoms connected to B site atom {b_site} is not 6 but {len(x_idx)}.")
        
        bx_raw = xpos_frame[x_idx, :] - bpos_frame[b_site, :]
        bx = octahedra_coords_into_bond_vectors(bx_raw, mymat)
        
        if rotated:
            order1 = match_bx_orthogonal_rotated(bx, rotmat)
        else:
            order1 = match_bx_orthogonal(bx)
        neigh_list[b_site, :] = np.array(x_idx)[order1]

    neigh_list = neigh_list.astype(int)
    return neigh_list


def fit_octahedral_network_defect_tol(
    bpos_frame: np.ndarray,
    xpos_frame: np.ndarray,
    r: np.ndarray,
    mymat: np.ndarray,
    fpg_val_bx: List[float],
    structure_type: int,
    dict_basis: Optional[Dict[str, Any]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Resolve octahedral connectivity with defect tolerance.
    
    Parameters
    ----------
    bpos_frame : np.ndarray
        Coordinate array of B-sites, shape (N_B, 3)
    xpos_frame : np.ndarray
        Coordinate array of X-sites, shape (N_X, 3)
    r : np.ndarray
        Distance matrix of B-X sites
    mymat : np.ndarray
        Lattice matrix, shape (3, 3)
    fpg_val_bx : list of float
        Defined B-X bond information
    structure_type : int
        Structure type indicating orientation and connectivity
    dict_basis : dict, optional
        Basis dictionary for structure_type 3
    
    Returns
    -------
    np.ndarray or tuple of (np.ndarray, np.ndarray)
        B-X connectivity matrix, optionally with reference initial matrices
    """
    from ...utils.geometry.structural_constants import DEFAULT_POPULATION_GAP_TOL
    
    try:
        max_bx_distance = find_population_gap(
            r, fpg_val_bx[0], fpg_val_bx[1], tol=DEFAULT_POPULATION_GAP_TOL)
        
        bxs = []
        bxc = []
        for b_site, x_list in enumerate(r):
            x_idx = [i for i in range(len(x_list)) if x_list[i] < max_bx_distance]
            bxc.append(len(x_idx))
            bxs.append(x_idx)
        bxc = np.array(bxc)
        
        if np.amax(bxc) != 6 or np.amin(bxc) != 6:
            ndef = np.sum(bxc != 6)
            if np.amax(bxc) < 8 and np.amin(bxc) > 4:
                logger.info(
                    f"Structure contains {ndef} out of {len(bxc)} octahedra with defects. "
                    f"Screened out of the population.")
            else:
                raise TypeError(
                    "Structure contains octahedra with complex defect configuration, "
                    "leading to unresolvable connectivity.")
    
    except (ValueError, TypeError):
        closebx = np.argsort(r, axis=0)[:2, :]
        binx, bcount = np.unique(closebx, return_counts=True)
        if (not np.array_equal(binx, np.arange(0, r.shape[0]))) or \
           (not np.array_equal(bcount, np.ones_like(bcount) * 6)):
            raise ValueError(
                "Can't separate the different neighbours. Check fpg_val values or "
                "if initial structure is defected or too distorted.")
        bxs = convert_xb_to_bx(closebx)
           
    neigh_list = np.zeros((bpos_frame.shape[0], 6))
    ref_initial = np.zeros((bpos_frame.shape[0], 3, 3))
    
    for b_site, bxcom in enumerate(bxs):
        if len(bxcom) == 6:
            try:
                bx_raw = xpos_frame[bxcom, :] - bpos_frame[b_site, :]
                bx = octahedra_coords_into_bond_vectors(bx_raw, mymat)
                
                if structure_type == 1:
                    order1 = match_bx_orthogonal(bx)
                    neigh_list[b_site, :] = np.array(bxcom)[order1].astype(int)
                elif structure_type == 2:
                    order1, _, _ = match_bx_arbitrary(bx, dict_basis or {})
                    neigh_list[b_site, :] = np.array(bxcom)[order1].astype(int)
                elif structure_type == 3:
                    if dict_basis is None:
                        raise ValueError("dict_basis required for structure_type 3")
                    order1, ref1, _ = match_bx_arbitrary(bx, dict_basis)
                    neigh_list[b_site, :] = np.array(bxcom)[order1].astype(int)
                    ref_initial[b_site, :] = ref1
            
            except ValueError:
                neigh_list[b_site, :] = np.nan
                if structure_type == 3:
                    ref_initial[b_site, :] = np.nan
        else:
            neigh_list[b_site, :] = np.nan
            if structure_type == 3:
                ref_initial[b_site, :] = np.nan
    
    if np.sum(np.isnan(neigh_list[:, 0])) > neigh_list.shape[0] * 0.1 and \
       np.sum(np.isnan(neigh_list[:, 0])) > 1:
        raise ValueError(
            f"There are {np.sum(np.isnan(neigh_list[:, 0]))} out of "
            f"{neigh_list.shape[0]} octahedra with unsolvable B-X connections.")
    
    if structure_type in (1, 2):
        return neigh_list
    else:
        return neigh_list, ref_initial


def fit_octahedral_network_defect_tol_non_orthogonal(
    bpos_frame: np.ndarray,
    xpos_frame: np.ndarray,
    r: np.ndarray,
    mymat: np.ndarray,
    fpg_val_bx: List[float],
    structure_type: int,
    rotmat: np.ndarray,
) -> np.ndarray:
    """Resolve octahedral connectivity with defect tolerance for non-orthogonal structures.
    
    Parameters
    ----------
    bpos_frame : np.ndarray
        Coordinate array of B-sites, shape (N_B, 3)
    xpos_frame : np.ndarray
        Coordinate array of X-sites, shape (N_X, 3)
    r : np.ndarray
        Distance matrix of B-X sites
    mymat : np.ndarray
        Lattice matrix, shape (3, 3)
    fpg_val_bx : list of float
        Defined B-X bond information
    structure_type : int
        Structure type indicating orientation and connectivity
    rotmat : np.ndarray
        Rotation matrix for alignment, shape (3, 3)
    
    Returns
    -------
    np.ndarray
        B-X connectivity matrix, shape (N_B, 6)
    """
    from ...utils.geometry.structural_constants import DEFAULT_POPULATION_GAP_TOL
    
    try:
        max_bx_distance = find_population_gap(
            r, fpg_val_bx[0], fpg_val_bx[1], tol=DEFAULT_POPULATION_GAP_TOL)
        
        bxs = []
        bxc = []
        for b_site, x_list in enumerate(r):
            x_idx = [i for i in range(len(x_list)) if x_list[i] < max_bx_distance]
            bxc.append(len(x_idx))
            bxs.append(x_idx)
        bxc = np.array(bxc)
        
        if np.amax(bxc) != 6 or np.amin(bxc) != 6:
            ndef = np.sum(bxc != 6)
            if np.amax(bxc) == 7 and np.amin(bxc) == 5:
                logger.info(
                    f"Structure contains {ndef} out of {len(bxc)} octahedra with defects. "
                    f"Screened out of the population.")
            else:
                raise TypeError(
                    "Structure contains octahedra with complex defect configuration, "
                    "leading to unresolvable connectivity.")
    
    except (ValueError, TypeError):
        closebx = np.argsort(r, axis=0)[:2, :]
        binx, bcount = np.unique(closebx, return_counts=True)
        if (not np.array_equal(binx, np.arange(0, r.shape[0]))) or \
           (not np.array_equal(bcount, np.ones_like(bcount) * 6)):
            raise ValueError(
                "Can't separate the different neighbours. Check fpg_val values or "
                "if initial structure is defected or too distorted.")
        logger.warning(
            "Detected relatively high deviation in B-X connectivity, used more aggressive fit.")
        bxs = convert_xb_to_bx(closebx)
        
    neigh_list = np.zeros((bpos_frame.shape[0], 6))
    for b_site, bxcom in enumerate(bxs):
        if len(bxcom) == 6:
            bx_raw = xpos_frame[bxcom, :] - bpos_frame[b_site, :]
            bx = octahedra_coords_into_bond_vectors(bx_raw, mymat)

            order1 = match_bx_orthogonal_rotated(bx, rotmat)
            neigh_list[b_site, :] = np.array(bxcom)[order1].astype(int)
        else:
            neigh_list[b_site, :] = np.nan

    return neigh_list


def find_polytype_network(
    bpos_frame: np.ndarray,
    xpos_frame: np.ndarray,
    r: np.ndarray,
    mymat: np.ndarray,
    neigh_list: np.ndarray,
    bb_search: float = 10.0,
) -> Tuple[List[str], List[np.ndarray], Dict[str, List[int]]]:
    """Resolve octahedral connectivity and output polytype information.
    
    Parameters
    ----------
    bpos_frame : np.ndarray
        Coordinate array of B-sites, shape (N_B, 3)
    xpos_frame : np.ndarray
        Coordinate array of X-sites, shape (N_X, 3)
    r : np.ndarray
        Distance matrix of B-B sites
    mymat : np.ndarray
        Lattice matrix, shape (3, 3)
    neigh_list : np.ndarray
        B-X connectivity matrix, shape (N_B, 6)
    bb_search : float
        Search radius for B-B neighbors
    
    Returns
    -------
    tuple of (list of str, list of np.ndarray, dict)
        Connection type strings, connectivity arrays, category dictionary
    """
    if np.isnan(neigh_list).any():
        raise TypeError(
            "Polytype recognition is not compatible with initial structure with defects.")
    
    def category_register(cat: Dict[str, List[int]], typestr: str, b: int) -> Dict[str, List[int]]:
        """Register octahedra into categories dict."""
        if typestr not in cat:
            cat[typestr] = []
        cat[typestr].append(b)
        return cat

    connectivity = []
    conntype_str = []
    conntype_str_all = []
    conn_category = {}
    
    for b0, b1_list in enumerate(r):
        b1 = [i for i in range(len(b1_list)) if (b1_list[i] < bb_search and i != b0)]
        if len(b1) == 0:
            raise ValueError(
                f"Can't find any other B-site around B-site {b0} within search radius "
                f"{bb_search} angstrom.")
            
        conn = np.empty((0, 2))
        for b1_idx in b1:
            intersect = list(set(neigh_list[b0, :]).intersection(neigh_list[b1_idx, :]))
            if len(intersect) > 0:
                if len(intersect) == 2:
                    bond_raw = xpos_frame[intersect, :] - bpos_frame[b0, :]
                    bond0 = apply_pbc_cart_vecs(bond_raw, mymat)
                    bond_raw = xpos_frame[intersect, :] - bpos_frame[b1_idx, :]
                    bond1 = apply_pbc_cart_vecs(bond_raw, mymat)
                    dots = np.sum(
                        (bond0 / np.linalg.norm(bond0, axis=1).reshape(2, 1)) *
                        (bond1 / np.linalg.norm(bond1, axis=1).reshape(2, 1)), axis=1)
                    if np.amax(dots) < -0.8:
                        intersect = [intersect[0]]
                conn = np.concatenate((conn, np.array([[b1_idx, len(intersect)]])).astype(int))
                
        if conn.shape[0] == 0:
            conntype_str.append("isolated")
            conntype_str_all.append("isolated")
            connectivity.append(conn)
            conn_category = category_register(conn_category, "isolated", b0)
        else:
            conntype = set(list(conn[:, 1]))
            if len(conntype - {1, 2, 3}) != 0:
                raise TypeError(f"Found an unexpected connectivity type {conntype}.")
            
            if len(conntype) == 1:
                if conntype == {1}:
                    conntype_str.append("corner")
                    conntype_str_all.append("corner")
                    conn_category = category_register(conn_category, "corner", b0)
                elif conntype == {2}:
                    conntype_str.append("edge")
                    conntype_str_all.append("edge")
                    conn_category = category_register(conn_category, "edge", b0)
                elif conntype == {3}:
                    conntype_str.append("face")
                    conntype_str_all.append("face")
                    conn_category = category_register(conn_category, "face", b0)
            else:
                strt = []
                for ct in conntype:
                    if ct == 1:
                        strt.append("corner")
                        conntype_str_all.append("corner")
                    elif ct == 2:
                        strt.append("edge")
                        conntype_str_all.append("edge")
                    elif ct == 3:
                        strt.append("face")
                        conntype_str_all.append("face")
                strt = "+".join(strt)
                conntype_str.append(strt)
                conn_category = category_register(conn_category, strt, b0)
                
            connectivity.append(conn)
        
    if len(set(conntype_str)) == 1:
        logger.info(f"Octahedral connectivity: {list(set(conntype_str))[0]}-sharing")
    else:
        conntypestr = "+".join(list(set(conntype_str_all)))
        logger.info(f"Octahedral connectivity: mixed - {conntypestr}")
        
    return conntype_str, connectivity, conn_category


def simply_calc_distortion(
    struct,
    neigh_list: np.ndarray,
    b_index: List[int],
    x_index: List[int],
    dict_basis: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute octahedral distortion of a structure.
    
    This function works on a single structure (not trajectories), making it
    suitable for structural characterization.
    
    Parameters
    ----------
    struct : pymatgen.Structure
        Structure to analyze
    neigh_list : np.ndarray
        Octahedra connectivity, shape (N_B, 6)
    b_index : list of int
        B-site atom indices
    x_index : list of int
        X-site atom indices
    dict_basis : dict
        Basis dictionary
    
    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Distortion amplitudes, standard deviations
    """
    from ...utils.geometry.structural_constants import DEFAULT_VOLUME_TOL
    
    bpos = struct.cart_coords[b_index, :]
    xpos = struct.cart_coords[x_index, :]
    
    mymat = struct.lattice.matrix
    
    dist_dim = len(dict_basis) - 3
    disto = np.empty((0, dist_dim))
    rmat = np.zeros((len(b_index), 3, 3))
    rmsd = np.zeros((len(b_index), 1))
    
    for b_site in range(len(b_index)):
        if np.isnan(neigh_list[b_site, :]).any():
            a = np.empty((1, dist_dim))
            a[:] = np.nan
            rmat[b_site, :] = np.nan
            rmsd[b_site] = np.nan
            disto = np.concatenate((disto, a), axis=0)
        else:
            raw = xpos[neigh_list[b_site, :].astype(int), :] - bpos[b_site, :]
            bx = octahedra_coords_into_bond_vectors(raw, mymat)
            dist_val, rotmat, rmsd_val = calc_distortions_from_bond_vectors_full(bx, dict_basis)
            rmat[b_site, :] = rotmat
            rmsd[b_site] = rmsd_val
            disto = np.concatenate((disto, dist_val.reshape(1, dist_dim)), axis=0)
    
    temp_var = np.var(disto, axis=0)
    temp_std = np.divide(np.sqrt(temp_var), np.nanmean(disto, axis=0))
    temp_dist = np.nanmean(disto, axis=0)
    
    for i in range(len(temp_dist)):
        if temp_dist[i] < DEFAULT_VOLUME_TOL:
            temp_dist[i] = 0
    
    return temp_dist, temp_std

