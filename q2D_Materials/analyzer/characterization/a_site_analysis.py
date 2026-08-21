"""A-site analysis functions for organic A-sites and mixed halide octahedra.

This module provides functions for analyzing and identifying organic A-site
molecules and mixed halide configurations in perovskite structures.

Parts of this code are from PDynA (https://github.com/WMD-group/PDynA):
MIT License - Copyright (c) 2022 Xia Liang
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from ...utils.geometry.structural_utils import apply_pbc_cart_vecs
from ...utils.geometry.structural_constants import DEFAULT_DISTINCT_THRESHOLD, DEFAULT_HBOND_MAX_DISTANCE


def centmass_organic(
    st0pos: np.ndarray,
    latmat: np.ndarray,
    env: List[List[int]],
) -> np.ndarray:
    """Find center of mass of organic A-site.
    
    Parameters
    ----------
    st0pos : np.ndarray
        Atomic positions, shape (N_atoms, 3)
    latmat : np.ndarray
        Lattice matrix, shape (3, 3)
    env : list of list of int
        Environment of A-site [C indices, N indices, H indices]
    
    Returns
    -------
    np.ndarray
        Center of mass positions, shape (3,)
    """
    c = env[0]
    n = env[1]
    h = env[2]
    
    refc = st0pos[[c[0]], :]
    cs = apply_pbc_cart_vecs(st0pos[c, :] - refc, latmat)
    ns = apply_pbc_cart_vecs(st0pos[n, :] - refc, latmat)
    hs = apply_pbc_cart_vecs(st0pos[h, :] - refc, latmat)
    
    mass = refc + (np.sum(cs, axis=0) * 12 + np.sum(ns, axis=0) * 14 +
                   np.sum(hs, axis=0) * 1) / (12 * len(c) + 14 * len(n) + 1 * len(h))

    return mass


def centmass_organic_vec(
    pos: np.ndarray,
    latmat: np.ndarray,
    env: List[List[int]],
) -> np.ndarray:
    """Find center of mass of organic A-site (vectorized version).
    
    Parameters
    ----------
    pos : np.ndarray
        Atomic positions, shape (N_frames, N_atoms, 3)
    latmat : np.ndarray
        Lattice matrices, shape (N_frames, 3, 3)
    env : list of list of int
        Environment of A-site [C indices, N indices, H indices]
    
    Returns
    -------
    np.ndarray
        Center of mass positions, shape (N_frames, 3)
    """
    c = env[0]
    n = env[1]
    h = env[2]
    
    refc = pos[:, [c[0]], :]
    cs = apply_pbc_cart_vecs(pos[:, c, :] - refc, latmat)
    ns = apply_pbc_cart_vecs(pos[:, n, :] - refc, latmat)
    hs = apply_pbc_cart_vecs(pos[:, h, :] - refc, latmat)
    
    mass = refc[:, 0, :] + (np.sum(cs, axis=1) * 12 + np.sum(ns, axis=1) * 14 +
                             np.sum(hs, axis=1) * 1) / (12 * len(c) + 14 * len(n) + 1 * len(h))

    return mass


def find_b_cage_and_disp(
    pos: np.ndarray,
    mymat: np.ndarray,
    cent: np.ndarray,
    bs: List[int],
) -> np.ndarray:
    """Find displacement of A-site with respect to capsulating B-X cage.
    
    Parameters
    ----------
    pos : np.ndarray
        Atomic positions, shape (N_frames, N_atoms, 3)
    mymat : np.ndarray
        Lattice matrices, shape (N_frames, 3, 3)
    cent : np.ndarray
        Center of mass of A-site, shape (N_frames, 3)
    bs : list of int
        Indices of B-site atoms
    
    Returns
    -------
    np.ndarray
        Relative displacement of A-site, shape (N_frames, 3)
    """
    b8pos = pos[:, bs, :]
    bx = b8pos - np.expand_dims(cent, axis=1)
    
    bx = apply_pbc_cart_vecs(bx, mymat)
    
    bs_pos = bx + np.expand_dims(cent, axis=1)
    xcent = np.mean(bs_pos, axis=1)
    
    disp = cent - xcent
    
    return disp


def match_mixed_halide_octa_dot(
    bx: np.ndarray,
    hals: List[str],
) -> Tuple[Tuple[int, int], int]:
    """Find configuration class in binary-mixed halide octahedron.
    
    Parameters
    ----------
    bx : np.ndarray
        Six B-X bond vectors, shape (6, 3)
    hals : list of str
        List of halide species symbols
    
    Returns
    -------
    tuple of (tuple of int, int)
        Configuration class and configuration number
    """
    ideal_coords = [
        [-1, 0, 0], [0, -1, 0], [0, 0, -1],
        [0, 0, 1], [0, 1, 0], [1, 0, 0]]
    
    brnum = hals.count('Br')
    
    if brnum == 0:
        return (0, 0), 0
    elif brnum == 1:
        return (1, 0), 1
    elif brnum == 2:
        brs = []
        for i, h in enumerate(hals):
            if h == "Br":
                brs.append(i)

        ang = np.dot(bx[brs[0], :], bx[brs[1], :])
        
        if ang < -0.7:
            return (2, 1), 3
        elif abs(ang) < 0.3:
            return (2, 0), 2
        else:
            raise ValueError(f"Can't distinguish local halide environment. Angular term: {ang}")
    
    elif brnum == 3:
        brs = []
        for i, h in enumerate(hals):
            if h == "Br":
                brs.append(i)

        ang = [np.dot(bx[brs[0], :], bx[brs[1], :]),
               np.dot(bx[brs[0], :], bx[brs[2], :]),
               np.dot(bx[brs[1], :], bx[brs[2], :])]
        
        if min(ang) < -0.7:
            return (3, 1), 5
        elif abs(min(ang)) < 0.3:
            return (3, 0), 4
        else:
            raise ValueError(f"Can't distinguish local halide environment. Angular terms: {ang}")
    
    elif brnum == 4:
        ios = []
        for i, h in enumerate(hals):
            if h == "I":
                ios.append(i)

        ang = np.dot(bx[ios[0], :], bx[ios[1], :])
        
        if ang < -0.7:
            return (4, 1), 7
        elif abs(ang) < 0.3:
            return (4, 0), 6
        else:
            raise ValueError(f"Can't distinguish local halide environment. Angular term: {ang}")

    elif brnum == 5:
        return (5, 0), 8
    elif brnum == 6:
        return (6, 0), 9
    else:
        raise ValueError(f"Invalid number of Br atoms: {brnum}")

