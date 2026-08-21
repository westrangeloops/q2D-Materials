"""PBC image precomputation for cavity tracing.

Pre-computes 27 (or fewer) periodic images of B/X atoms and provides
nearest-neighbor lookup for efficient cavity detection.
"""

import sys
import numpy as np
from typing import List, Tuple, Optional, Union


def _precompute_27_images(
    atom_positions: np.ndarray,
    atom_indices: np.ndarray,
    cell: np.ndarray,
    pbc: Union[bool, List[bool]] = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute all 27 PBC images for atoms.
    
    This function generates all 27 periodic images (3×3×3) for each atom
    and stores them for efficient nearest-neighbor searches.
    
    Parameters
    ----------
    atom_positions : np.ndarray
        Atom positions (M, 3) - can be any coordinates (wrapped or unwrapped)
    atom_indices : np.ndarray
        Atom indices (M,)
    cell : np.ndarray
        Unit cell matrix (3, 3)
    pbc : bool or list of bool, default=True
        Periodic boundary conditions
        
    Returns
    -------
    tuple
        (all_image_positions, all_image_indices, all_image_labels)
        - all_image_positions: (M * n_images, 3) - all 27 images for all atoms
        - all_image_indices: (M * n_images,) - atom index for each image
        - all_image_labels: (M * n_images, 3) - PBC image label (i, j, k) for each image
    """
    # Parse PBC axes
    if isinstance(pbc, bool):
        pbc_axes = np.array([1, 1, 1] if pbc else [0, 0, 0], dtype=np.int32)
    else:
        pbc_axes = np.array([int(p) for p in pbc], dtype=np.int32)
    
    # Generate all image offsets based on PBC axes
    image_offsets = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                # Skip images for non-periodic axes
                if i != 0 and pbc_axes[0] == 0:
                    continue
                if j != 0 and pbc_axes[1] == 0:
                    continue
                if k != 0 and pbc_axes[2] == 0:
                    continue
                image_offsets.append([i, j, k])
    
    image_offsets = np.array(image_offsets, dtype=np.float64)
    n_images = len(image_offsets)
    n_atoms = len(atom_indices)
    
    # Verify we have exactly 27 images (or 9 for 2D, 3 for 1D, 1 for 0D)
    expected_images = np.prod([3 if ax else 1 for ax in pbc_axes])
    if n_images != expected_images:
        print(f"    WARNING: Expected {expected_images} images but got {n_images}", file=sys.stderr)
    
    # Pre-compute all images
    all_image_positions = []
    all_image_indices = []
    all_image_labels = []
    
    for atom_idx, pos in zip(atom_indices, atom_positions):
        for img_offset in image_offsets:
            # Translate position by image offset: pos + (i, j, k) @ cell
            # This extends the space in all 27 directions
            translated = pos + img_offset @ cell
            all_image_positions.append(translated)
            all_image_indices.append(atom_idx)
            all_image_labels.append(img_offset)
    
    # Verify we created the expected number of images
    expected_total = n_atoms * n_images
    actual_total = len(all_image_positions)
    if actual_total != expected_total:
        print(f"    ERROR: Expected {expected_total} total images but got {actual_total}", file=sys.stderr)
        print(f"    n_atoms={n_atoms}, n_images={n_images}", file=sys.stderr)
    
    return (
        np.array(all_image_positions),
        np.array(all_image_indices, dtype=np.int32),
        np.array(all_image_labels, dtype=np.int32)
    )


def _find_nearest_from_cached_images(
    reference_position: np.ndarray,
    cached_image_positions: np.ndarray,
    cached_image_indices: np.ndarray,
    cached_image_labels: np.ndarray,
    n_neighbors: int,
    exclude_indices: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find N nearest positions from pre-computed 27-image cache.
    
    Uses Euclidean distance which is correct for any coordinate system
    (positive or negative values). The distance is the magnitude of the
    vector difference: ||pos - reference||
    
    Parameters
    ----------
    reference_position : np.ndarray
        Center position to measure distances from. Shape: (3,).
    cached_image_positions : np.ndarray
        Pre-computed image positions. Shape: (M * n_images, 3).
    cached_image_indices : np.ndarray
        Atom indices for each image. Shape: (M * n_images,).
    cached_image_labels : np.ndarray
        Image labels for each position. Shape: (M * n_images, 3).
    n_neighbors : int
        Number of nearest positions to return.
    exclude_indices : np.ndarray or None, default=None
        Atom indices to exclude from search.
        
    Returns
    -------
    tuple
        (nearest_indices, nearest_positions, nearest_distances, image_labels)
    """
    # Create exclusion set for fast lookup
    exclude_set = set(exclude_indices) if exclude_indices is not None else set()

    # Calculate Euclidean distances for all images
    # This is correct regardless of coordinate sign: ||a - b|| = sqrt(sum((a_i - b_i)^2))
    distances = np.linalg.norm(cached_image_positions - reference_position, axis=1)
    
    # Debug: verify distance calculation
    if len(distances) == 0:
        print(f"    ERROR: No distances calculated! cached_image_positions shape: {cached_image_positions.shape}", file=sys.stderr)
        return (np.array([], dtype=np.int32), np.array([]), np.array([]), np.array([], dtype=np.int32))
    
    # Filter out excluded atoms and create list of (distance, idx, pos, label)
    all_data = []
    for i in range(len(cached_image_positions)):
        atom_idx = cached_image_indices[i]
        if atom_idx in exclude_set:
            continue
        all_data.append((
            distances[i],
            atom_idx,
            cached_image_positions[i],
            cached_image_labels[i]
        ))
    
    if len(all_data) == 0:
        print(f"    ERROR: No valid candidates after filtering! exclude_set={exclude_set}", file=sys.stderr)
        return (np.array([], dtype=np.int32), np.array([]), np.array([]), np.array([], dtype=np.int32))
    
    # Sort by distance
    all_data.sort(key=lambda x: x[0])
    
    # Select N nearest
    n_return = min(n_neighbors, len(all_data))
    nearest_data = all_data[:n_return]
    
    # Debug: print distances of selected atoms
    print(f"    Selected {n_return} nearest atoms with distances: {[f'{d[0]:.3f}' for d in nearest_data]}", file=sys.stderr)
    print(f"    Selected atom indices: {[d[1] for d in nearest_data]}", file=sys.stderr)
    print(f"    Selected image labels: {[tuple(d[3]) for d in nearest_data]}", file=sys.stderr)
    
    # Extract arrays
    nearest_distances = np.array([d[0] for d in nearest_data])
    nearest_indices = np.array([d[1] for d in nearest_data], dtype=np.int32)
    nearest_positions = np.array([d[2] for d in nearest_data])
    image_labels = np.array([d[3] for d in nearest_data], dtype=np.int32)
    
    return nearest_indices, nearest_positions, nearest_distances, image_labels
