"""Basic structural utilities for PBC and coordinate conversion.

Parts of this code are from PDynA (https://github.com/WMD-group/PDynA):
MIT License - Copyright (c) 2022 Xia Liang
"""

from typing import Tuple, Optional
import numpy as np


def get_cart_from_frac(
    frac: np.ndarray,
    latmat: np.ndarray,
) -> np.ndarray:
    """Convert fractional to cartesian coordinates.
    
    Parameters
    ----------
    frac : np.ndarray
        Fractional coordinates, shape (N, 3) or (N_frames, N, 3)
    latmat : np.ndarray
        Lattice matrix, shape (3, 3) or (N_frames, 3, 3)
    
    Returns
    -------
    np.ndarray
        Cartesian coordinates, same shape as frac
    """
    if frac.ndim != latmat.ndim:
        raise ValueError("The dimension of the input arrays do not match.")
    if frac.shape[-1] != 3 or latmat.shape[-2:] != (3, 3):
        raise TypeError("Must be 3D vectors.")
        
    if frac.ndim == 2:
        pass
    elif frac.ndim == 3:
        if frac.shape[0] != latmat.shape[0]:
            raise ValueError("The frame number of input arrays do not match.")
    else:
        raise TypeError("Can only deal with 2 or 3D arrays.")
            
    return np.matmul(frac, latmat)


def get_frac_from_cart(
    cart: np.ndarray,
    latmat: np.ndarray,
) -> np.ndarray:
    """Convert cartesian to fractional coordinates.
    
    Parameters
    ----------
    cart : np.ndarray
        Cartesian coordinates, shape (N, 3) or (N_frames, N, 3)
    latmat : np.ndarray
        Lattice matrix, shape (3, 3) or (N_frames, 3, 3)
    
    Returns
    -------
    np.ndarray
        Fractional coordinates, same shape as cart
    """
    if cart.ndim != latmat.ndim:
        raise ValueError("The dimension of the input arrays do not match.")
    if cart.shape[-1] != 3 or latmat.shape[-2:] != (3, 3):
        raise TypeError("Must be 3D vectors.")
        
    if cart.ndim == 2:
        pass
    elif cart.ndim == 3:
        if cart.shape[0] != latmat.shape[0]:
            raise ValueError("The frame number of input arrays do not match.")
    else:
        raise TypeError("Can only deal with 2 or 3D arrays.")
            
    return np.matmul(cart, np.linalg.inv(latmat))


def apply_pbc_cart_vecs(
    vecs: np.ndarray,
    mymat: np.ndarray,
) -> np.ndarray:
    """Apply PBC to vectors using lattice matrix.
    
    Parameters
    ----------
    vecs : np.ndarray
        Vectors to wrap, shape (N, 3) or (N_frames, N, 3)
    mymat : np.ndarray
        Lattice matrix, shape (3, 3) or (N_frames, 3, 3)
    
    Returns
    -------
    np.ndarray
        Vectors with PBC applied, same shape as input
    """
    vecs_frac = get_frac_from_cart(vecs, mymat)
    vecs_pbc = get_cart_from_frac(vecs_frac - np.round(vecs_frac), mymat)
    return vecs_pbc


def apply_pbc_cart_vecs_single_frame(
    vecs: np.ndarray,
    mymat: np.ndarray,
) -> np.ndarray:
    """Apply PBC to vectors for a single frame.
    
    Parameters
    ----------
    vecs : np.ndarray
        Vectors to wrap, shape (N, 3)
    mymat : np.ndarray
        Lattice matrix, shape (3, 3)
    
    Returns
    -------
    np.ndarray
        Vectors with PBC applied, same shape as input
    """
    vecs_frac = np.matmul(vecs, np.linalg.inv(mymat))
    vecs_pbc = np.matmul(vecs_frac - np.round(vecs_frac), mymat)
    return vecs_pbc


def periodicity_fold(
    arrin: np.ndarray,
    n_fold: int = 4,
) -> np.ndarray:
    """Fold angles to handle periodicity.
    
    Parameters
    ----------
    arrin : np.ndarray
        Input array of angles
    n_fold : int
        Fold period (4, 2, or 8)
    
    Returns
    -------
    np.ndarray
        Folded angles
    """
    from copy import deepcopy
    arr = deepcopy(arrin)
    if n_fold == 4:
        arr[arr < -45] = arr[arr < -45] + 90
        arr[arr < -45] = arr[arr < -45] + 90
        arr[arr > 45] = arr[arr > 45] - 90
        arr[arr > 45] = arr[arr > 45] - 90
    elif n_fold == 2:
        arr[arr < -90] = arr[arr < -90] + 180
        arr[arr > 90] = arr[arr > 90] - 180
    elif n_fold == 8:
        arr[arr < -45] = arr[arr < -45] + 90
        arr[arr < -45] = arr[arr < -45] + 90
        arr[arr > 45] = arr[arr > 45] - 90
        arr[arr > 45] = arr[arr > 45] - 90
        arr = np.abs(arr)
    return arr


def get_volume(lattice: np.ndarray) -> float:
    """Calculate volume from lattice parameters.
    
    Parameters
    ----------
    lattice : np.ndarray
        Lattice parameters, shape (6,) or (N, 6) [A, B, C, alpha, beta, gamma]
    
    Returns
    -------
    float or np.ndarray
        Volume(s) of the lattice
    """
    if lattice.shape == (6,):
        A, B, C, a, b, c = lattice
        a = a / 180 * np.pi
        b = b / 180 * np.pi
        c = c / 180 * np.pi
        vol = A * B * C * (1 - np.cos(a)**2 - np.cos(b)**2 - np.cos(c)**2 + 
                           2 * np.cos(a) * np.cos(b) * np.cos(c))**0.5
        return vol
    elif lattice.ndim == 2 and lattice.shape[1] == 6:
        vol_list = []
        for i in range(lattice.shape[0]):
            A, B, C, a, b, c = lattice[i, :]
            a = a / 180 * np.pi
            b = b / 180 * np.pi
            c = c / 180 * np.pi
            vol = A * B * C * (1 - np.cos(a)**2 - np.cos(b)**2 - np.cos(c)**2 + 
                               2 * np.cos(a) * np.cos(b) * np.cos(c))**0.5
            vol_list.append(vol)
        return np.array(vol_list)
    else:
        raise TypeError("The input must have shape (6,) or (N, 6).")


def distance_matrix(
    v1: np.ndarray,
    v2: np.ndarray,
    latmat: np.ndarray,
    get_vec: bool = False,
) -> np.ndarray:
    """Compute distance matrix using matrix multiplication method.
    
    Should only be used for cases with alpha, beta, gamma angles all close to 90 degrees.
    
    Parameters
    ----------
    v1 : np.ndarray
        First set of coordinate vectors, shape (N1, 3)
    v2 : np.ndarray
        Second set of coordinate vectors, shape (N2, 3)
    latmat : np.ndarray
        Lattice matrix, shape (3, 3)
    get_vec : bool
        Whether to return distance vectors
    
    Returns
    -------
    np.ndarray
        Distance matrix, shape (N1, N2), or vectors if get_vec=True
    """
    dprec = np.float32
    
    f1 = get_frac_from_cart(v1, latmat)[:, np.newaxis, :].astype(dprec)
    f2 = get_frac_from_cart(v2, latmat)[np.newaxis, :, :].astype(dprec)
    
    df = np.repeat(f1, f2.shape[1], axis=1) - np.repeat(f2, f1.shape[0], axis=0)
    df = df - np.round(df)
    df = np.matmul(df, latmat.astype(dprec))
    
    if get_vec:
        return df
    else:
        return np.linalg.norm(df, axis=2)


def distance_matrix_ase(
    v1: np.ndarray,
    v2: np.ndarray,
    asecell,
    pbc: list,
    get_vec: bool = False,
) -> np.ndarray:
    """Compute distance matrix using ASE algorithm.
    
    Parameters
    ----------
    v1 : np.ndarray
        First set of coordinate vectors, shape (N1, 3)
    v2 : np.ndarray
        Second set of coordinate vectors, shape (N2, 3)
    asecell : ase.Atoms
        Atoms object of the system
    pbc : list of bool
        Periodic boundary conditions in three directions
    get_vec : bool
        Whether to return distance vectors
    
    Returns
    -------
    np.ndarray
        Distance matrix, shape (N1, N2), or vectors if get_vec=True
    """
    from ase.geometry import get_distances
    
    D, D_len = get_distances(v1, v2, cell=asecell, pbc=pbc)
    
    if get_vec:
        return D
    else:
        return D_len


def distance_matrix_ase_replace(
    v1: np.ndarray,
    v2: np.ndarray,
    asecell,
    newcell: np.ndarray,
    pbc: list,
    get_vec: bool = False,
) -> np.ndarray:
    """Compute distance matrix using ASE with a new cell matrix.
    
    Parameters
    ----------
    v1 : np.ndarray
        First set of coordinate vectors, shape (N1, 3)
    v2 : np.ndarray
        Second set of coordinate vectors, shape (N2, 3)
    asecell : ase.Atoms
        Atoms object of the system
    newcell : np.ndarray
        New cell matrix, shape (3, 3)
    pbc : list of bool
        Periodic boundary conditions in three directions
    get_vec : bool
        Whether to return distance vectors
    
    Returns
    -------
    np.ndarray
        Distance matrix, shape (N1, N2), or vectors if get_vec=True
    """
    from ase.geometry import get_distances
    
    celltemp = asecell.copy()
    celltemp.array = newcell
    
    D, D_len = get_distances(v1, v2, cell=celltemp, pbc=pbc)
    
    if get_vec:
        return D
    else:
        return D_len


def distance_matrix_handler(
    v1: np.ndarray,
    v2: np.ndarray,
    latmat: np.ndarray,
    asecell: Optional[object] = None,
    pbc: Optional[list] = None,
    complex_pbc: bool = False,
    replace: bool = True,
    get_vec: bool = False,
) -> np.ndarray:
    """Compute distance matrix with different methods depending on requirement.
    
    Parameters
    ----------
    v1 : np.ndarray
        First set of coordinate vectors, shape (N1, 3) or (3,)
    v2 : np.ndarray
        Second set of coordinate vectors, shape (N2, 3) or (3,)
    latmat : np.ndarray
        Lattice matrix, shape (3, 3)
    asecell : ase.Atoms, optional
        Atoms object of the system
    pbc : list of bool, optional
        Periodic boundary conditions in three directions
    complex_pbc : bool
        Whether to use the ASE algorithm
    replace : bool
        Whether to use a new cell matrix
    get_vec : bool
        Whether to return distance vectors
    
    Returns
    -------
    np.ndarray
        Distance matrix, shape (N1, N2), or vectors if get_vec=True
    """
    if v1.shape == (3,):
        v1 = v1[np.newaxis, :]
    if v2.shape == (3,):
        v2 = v2[np.newaxis, :]
    if v1.ndim != 2 or v1.shape[1] != 3 or v2.ndim != 2 or v2.shape[1] != 3:
        raise TypeError("The input arrays must be in shape (N*3) or (3,).")
    
    if complex_pbc is False:
        r = distance_matrix(v1, v2, latmat, get_vec=get_vec)
    else:
        if replace is False:
            r = distance_matrix_ase(v1, v2, asecell, pbc, get_vec=get_vec)
        else:
            r = distance_matrix_ase_replace(v1, v2, asecell, latmat, pbc, get_vec=get_vec)
    
    return r

