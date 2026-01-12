"""Resolve octahedra distortions and tilting from trajectory coordinates (MD simulations).

Parts of this code are from PDynA (https://github.com/WMD-group/PDynA):
MIT License - Copyright (c) 2022 Xia Liang
"""

from typing import Optional, Tuple, List, Dict, Any
import logging
import numpy as np
from scipy.spatial.transform import Rotation as sstr
from tqdm import tqdm
from joblib import Parallel, delayed
import contextlib

from .octahedral_analysis import (
    octahedra_coords_into_bond_vectors,
    calc_distortions_from_bond_vectors_full,
)
from ...utils.geometry.structural_utils import (
    distance_matrix_handler,
    apply_pbc_cart_vecs,
)
from ...utils.geometry.structural_constants import DEFAULT_RMSD_THRESHOLD

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Support progress bar within joblib parallelized computation."""
    class TqdmBatchCompletionCallback:
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    import joblib
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def _compute_frame_distortions(
    bpos: np.ndarray,
    xpos: np.ndarray,
    b_site: int,
    neigh_list: np.ndarray,
    mymat: np.ndarray,
    ref_initial: Optional[np.ndarray],
    rtr: Optional[np.ndarray],
    orthogonal_frame: bool,
    dict_basis: Dict[str, Any],
    dist_dim: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute distortions for a single octahedron in a frame.
    
    Parameters
    ----------
    bpos : np.ndarray
        B-site positions for frame, shape (N_B, 3)
    xpos : np.ndarray
        X-site positions for frame, shape (N_X, 3)
    b_site : int
        B-site index
    neigh_list : np.ndarray
        B-X connectivity matrix, shape (N_B, 6)
    mymat : np.ndarray
        Lattice matrix, shape (3, 3)
    ref_initial : np.ndarray, optional
        Individual reference for each octahedron
    rtr : np.ndarray, optional
        Rotation matrix from orthogonal directions
    orthogonal_frame : bool
        If structure is 3D perovskite aligned in orthogonal directions
    dict_basis : dict
        Basis dictionary
    dist_dim : int
        Distortion dimension
    
    Returns
    -------
    tuple of (np.ndarray, np.ndarray, float)
        Distortion values, rotation matrix, RMSD
    """
    if np.isnan(neigh_list[b_site, :]).any():
        dist_val = np.empty((1, dist_dim))
        dist_val[:] = np.nan
        rotmat = np.nan * np.ones((3, 3))
        rmsd = np.nan
        return dist_val, rotmat, rmsd
    
    raw = xpos[neigh_list[b_site, :].astype(int), :] - bpos[b_site, :]
    bx = octahedra_coords_into_bond_vectors(raw, mymat)
    
    if not orthogonal_frame:
        bx = np.matmul(bx, ref_initial[b_site, :])
    if rtr is not None:
        bx = np.matmul(bx, rtr)
  
    dist_val, rotmat, rmsd = calc_distortions_from_bond_vectors_full(bx, dict_basis)
    
    return dist_val.reshape(1, dist_dim), rotmat, rmsd


def _process_single_frame(
    fr: int,
    bpos: np.ndarray,
    xpos: np.ndarray,
    neigh_list: np.ndarray,
    latmat: np.ndarray,
    ref_initial: Optional[np.ndarray],
    rtr: Optional[np.ndarray],
    orthogonal_frame: bool,
    dict_basis: Dict[str, Any],
    dist_dim: int,
    bcount: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute octahedral tilting and distortion for a single frame.
    
    Parameters
    ----------
    fr : int
        Frame number
    bpos : np.ndarray
        B-site positions, shape (N_frames, N_B, 3)
    xpos : np.ndarray
        X-site positions, shape (N_frames, N_X, 3)
    neigh_list : np.ndarray
        B-X connectivity matrix, shape (N_B, 6)
    latmat : np.ndarray
        Lattice matrices, shape (N_frames, 3, 3)
    ref_initial : np.ndarray, optional
        Individual reference for each octahedron
    rtr : np.ndarray, optional
        Rotation matrix from orthogonal directions
    orthogonal_frame : bool
        If structure is 3D perovskite aligned in orthogonal directions
    dict_basis : dict
        Basis dictionary
    dist_dim : int
        Distortion dimension
    bcount : int
        Number of B-sites
    
    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        Distortions, tilts, RMSD values
    """
    mymat = latmat[fr, :]
    
    disto = np.empty((0, dist_dim))
    rmat = np.zeros((bcount, 3, 3))
    rmsd = np.zeros((bcount, 1))
    
    for b_site in range(bcount):
        dist_val, rotmat, rmsd_val = _compute_frame_distortions(
            bpos[fr, :], xpos[fr, :], b_site, neigh_list, mymat,
            ref_initial, rtr, orthogonal_frame, dict_basis, dist_dim)
        
        rmat[b_site, :] = rotmat
        rmsd[b_site] = rmsd_val
        disto = np.concatenate((disto, dist_val), axis=0)
    
    tilts = np.zeros((bcount, 3))
    for i in range(rmat.shape[0]):
        if np.isnan(neigh_list[i, :]).any():
            tilts[i, :] = np.nan
        else:
            tilts[i, :] = sstr.from_matrix(rmat[i, :]).as_euler('xyz', degrees=True)

    return (disto.reshape(1, bcount, dist_dim), tilts.reshape(1, bcount, 3), rmsd)


def _refit_octahedral_network(
    fr: int,
    bpos: np.ndarray,
    xpos: np.ndarray,
    latmat: np.ndarray,
    at0: Any,
    fpg_val_bx: List[float],
    structure_type: int,
    orthogonal_frame: bool,
    rtr: Optional[np.ndarray],
    complex_pbc: bool,
    dict_basis: Optional[Dict[str, Any]],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Refit octahedral network if large distortion is found.
    
    Parameters
    ----------
    fr : int
        Frame number
    bpos : np.ndarray
        B-site positions, shape (N_frames, N_B, 3)
    xpos : np.ndarray
        X-site positions, shape (N_frames, N_X, 3)
    latmat : np.ndarray
        Lattice matrices, shape (N_frames, 3, 3)
    at0 : object
        ASE Atoms object of initial frame
    fpg_val_bx : list of float
        Defined B-X bond information
    structure_type : int
        Structure type
    orthogonal_frame : bool
        If structure is 3D perovskite aligned in orthogonal directions
    rtr : np.ndarray, optional
        Rotation matrix from orthogonal directions
    complex_pbc : bool
        If cell has strong tilts
    dict_basis : dict, optional
        Basis dictionary
    
    Returns
    -------
    tuple of (np.ndarray, np.ndarray or None)
        New neighbor list, optionally new ref_initial
    """
    from .octahedral_analysis import (
        fit_octahedral_network_defect_tol,
        fit_octahedral_network_defect_tol_non_orthogonal,
    )
    
    bpos_frame = bpos[fr, :]
    xpos_frame = xpos[fr, :]
    rt = distance_matrix_handler(
        bpos_frame, xpos_frame, latmat[fr, :], at0.cell, at0.pbc, complex_pbc)
    
    mymat = latmat[fr, :]
    
    if orthogonal_frame:
        if rtr is None:
            neigh_list = fit_octahedral_network_defect_tol(
                bpos_frame, xpos_frame, rt, mymat, fpg_val_bx, structure_type, dict_basis)
        else:
            neigh_list = fit_octahedral_network_defect_tol_non_orthogonal(
                bpos_frame, xpos_frame, rt, mymat, fpg_val_bx, structure_type, rtr)
        return neigh_list, None
    else:
        result = fit_octahedral_network_defect_tol(
            bpos_frame, xpos_frame, rt, mymat, fpg_val_bx, structure_type, dict_basis)
        if isinstance(result, tuple):
            return result
        else:
            return result, None


def resolve_octahedra(
    bpos: np.ndarray,
    xpos: np.ndarray,
    readfr: List[int],
    at0: Any,
    enable_refit: bool,
    multi_thread: int,
    latmat: np.ndarray,
    fpg_val_bx: List[float],
    neigh_list: np.ndarray,
    orthogonal_frame: bool,
    structure_type: int,
    complex_pbc: bool,
    dict_basis: Dict[str, Any],
    dist_dim: int,
    ref_initial: Optional[np.ndarray] = None,
    rtr: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute octahedral tilting and distortion from trajectory coordinates.
    
    Parameters
    ----------
    bpos : np.ndarray
        B-site coordinates, shape (N_frames, N_B, 3)
    xpos : np.ndarray
        X-site coordinates, shape (N_frames, N_X, 3)
    readfr : list of int
        List of frame numbers to compute
    at0 : object
        ASE Atoms object of initial frame
    enable_refit : bool
        Enable structure refitting during computation if large distortion found
    multi_thread : int
        Number of threads for parallel processing (1 = disabled)
    latmat : np.ndarray
        Lattice matrices, shape (N_frames, 3, 3)
    fpg_val_bx : list of float
        Defined B-X bond information
    neigh_list : np.ndarray
        B-X connectivity matrix, shape (N_B, 6)
    orthogonal_frame : bool
        If structure is 3D perovskite aligned in orthogonal directions
    structure_type : int
        Structure type indicating orientation and connectivity
    complex_pbc : bool
        If cell has strong tilts
    dict_basis : dict
        Basis dictionary
    dist_dim : int
        Distortion dimension
    ref_initial : np.ndarray, optional
        Individual reference for each octahedron
    rtr : np.ndarray, optional
        Rotation matrix from orthogonal directions
    
    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        Distortions, tilts, refits record
    """
    if (ref_initial is not None) and (rtr is not None):
        raise TypeError(
            "Individual reference of octahedra and orthogonal lattice alignment "
            "cannot be simultaneously enabled.")

    bcount = bpos.shape[1]
    refits = np.empty((0, 2))
    
    if multi_thread == 1:
        di = np.empty((len(readfr), bcount, dist_dim))
        t = np.empty((len(readfr), bcount, 3))
        
        for subfr in tqdm(range(len(readfr))):
            fr = readfr[subfr]
            mymat = latmat[fr, :]
            
            disto = np.empty((0, dist_dim))
            rmat = np.zeros((bcount, 3, 3))
            rmsd = np.zeros((bcount, 1))
            
            for b_site in range(bcount):
                dist_val, rotmat, rmsd_val = _compute_frame_distortions(
                    bpos[fr, :], xpos[fr, :], b_site, neigh_list, mymat,
                    ref_initial, rtr, orthogonal_frame, dict_basis, dist_dim)
                
                rmat[b_site, :] = rotmat
                rmsd[b_site] = rmsd_val
                disto = np.concatenate((disto, dist_val), axis=0)
            
            if enable_refit and fr > 0 and max(np.amax(disto)) > 1:
                disto_prev = np.nanmean(disto, axis=0)
                neigh_list_prev = neigh_list
                
                neigh_list, ref_initial = _refit_octahedral_network(
                    fr, bpos, xpos, latmat, at0, fpg_val_bx, structure_type,
                    orthogonal_frame, rtr, complex_pbc, dict_basis)
                
                mymat = latmat[fr, :]
                
                disto = np.empty((0, dist_dim))
                rmat = np.zeros((bcount, 3, 3))
                rmsd = np.zeros((bcount, 1))
                
                for b_site in range(bcount):
                    dist_val, rotmat, rmsd_val = _compute_frame_distortions(
                        bpos[fr, :], xpos[fr, :], b_site, neigh_list, mymat,
                        ref_initial, rtr, orthogonal_frame, dict_basis, dist_dim)
                    
                    rmat[b_site, :] = rotmat
                    rmsd[b_site] = rmsd_val
                    disto = np.concatenate((disto, dist_val), axis=0)
                
                if (np.array_equal(np.nanmean(disto, axis=0), disto_prev) and
                    np.array_equal(neigh_list, neigh_list_prev)):
                    refits = np.concatenate((refits, np.array([[fr, 0]])), axis=0)
                else:
                    refits = np.concatenate((refits, np.array([[fr, 1]])), axis=0)
            
            di[subfr, :, :] = disto.reshape(1, bcount, dist_dim)
            
            tilts = np.zeros((bcount, 3))
            for i in range(rmat.shape[0]):
                tilts[i, :] = sstr.from_matrix(rmat[i, :]).as_euler('xyz', degrees=True)
            t[subfr, :, :] = tilts.reshape(1, bcount, 3)
            
    elif multi_thread > 1:
        di = np.empty((len(readfr), bcount, dist_dim))
        t = np.empty((len(readfr), bcount, 3))
        
        with tqdm_joblib(tqdm(desc="Progress", total=len(readfr))) as progress_bar:
            results = Parallel(n_jobs=multi_thread)(
                delayed(_process_single_frame)(
                    fr, bpos, xpos, neigh_list, latmat, ref_initial, rtr,
                    orthogonal_frame, dict_basis, dist_dim, bcount)
                for fr in readfr)
            
        assert len(results) == len(readfr)
        for fr_idx, each in enumerate(results):
            di[fr_idx, :, :] = each[0]
            t[fr_idx, :, :] = each[1]
    else:
        raise ValueError("The input multi-threading count must be a positive integer.")
        
    return di, t, refits

