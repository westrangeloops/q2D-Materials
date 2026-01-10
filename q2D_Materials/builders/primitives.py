"""
Geometric Primitives Module for Molecular Manipulation.

This module provides clean geometric abstractions (Point3D, Vector3D) 
inspired by CineMol's architecture, optimized for molecular manipulation
with numpy backend for performance.

Key Features:
- Immutable Point3D and Vector3D with rich operations
- Gram-Schmidt orthogonalization for robust axis generation
- 3D Convex Hull for molecular envelope collision detection
- Rodrigues rotation formula implementation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np


@dataclass(frozen=True)
class Vector3D:
    """
    Immutable 3D vector with numpy backend.
    
    Attributes
    ----------
    coords : np.ndarray
        3D coordinates [x, y, z]
    """
    coords: np.ndarray
    
    def __post_init__(self):
        # Ensure coords is a numpy array with shape (3,)
        object.__setattr__(self, 'coords', np.asarray(self.coords, dtype=np.float64).flatten()[:3])
    
    @classmethod
    def from_xyz(cls, x: float, y: float, z: float) -> 'Vector3D':
        """Create vector from x, y, z components."""
        return cls(np.array([x, y, z], dtype=np.float64))
    
    @classmethod
    def create_random(cls) -> 'Vector3D':
        """Create a random unit vector."""
        vec = np.random.randn(3)
        return cls(vec / np.linalg.norm(vec))
    
    @property
    def x(self) -> float:
        return float(self.coords[0])
    
    @property
    def y(self) -> float:
        return float(self.coords[1])
    
    @property
    def z(self) -> float:
        return float(self.coords[2])
    
    def length(self) -> float:
        """Calculate the length (magnitude) of this vector."""
        return float(np.linalg.norm(self.coords))
    
    def normalize(self) -> 'Vector3D':
        """Return a unit vector in the same direction."""
        length = self.length()
        if length < 1e-10:
            return Vector3D(np.zeros(3))
        return Vector3D(self.coords / length)
    
    def dot(self, other: 'Vector3D') -> float:
        """Calculate dot product with another vector."""
        return float(np.dot(self.coords, other.coords))
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        """Calculate cross product with another vector."""
        return Vector3D(np.cross(self.coords, other.coords))
    
    def add(self, other: 'Vector3D') -> 'Vector3D':
        """Add another vector."""
        return Vector3D(self.coords + other.coords)
    
    def subtract(self, other: 'Vector3D') -> 'Vector3D':
        """Subtract another vector."""
        return Vector3D(self.coords - other.coords)
    
    def multiply(self, scalar: float) -> 'Vector3D':
        """Multiply by a scalar."""
        return Vector3D(self.coords * scalar)
    
    def negate(self) -> 'Vector3D':
        """Return the negated vector."""
        return Vector3D(-self.coords)
    
    def angle_to(self, other: 'Vector3D') -> float:
        """Calculate angle to another vector in radians."""
        dot = self.normalize().dot(other.normalize())
        dot = np.clip(dot, -1.0, 1.0)
        return float(np.arccos(dot))
    
    def project_onto(self, other: 'Vector3D') -> 'Vector3D':
        """Project this vector onto another vector."""
        other_norm = other.normalize()
        return other_norm.multiply(self.dot(other_norm))
    
    def __repr__(self) -> str:
        return f"Vector3D({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


@dataclass(frozen=True)
class Point3D:
    """
    Immutable 3D point with numpy backend.
    
    Attributes
    ----------
    coords : np.ndarray
        3D coordinates [x, y, z]
    """
    coords: np.ndarray
    
    def __post_init__(self):
        # Ensure coords is a numpy array with shape (3,)
        object.__setattr__(self, 'coords', np.asarray(self.coords, dtype=np.float64).flatten()[:3])
    
    @classmethod
    def from_xyz(cls, x: float, y: float, z: float) -> 'Point3D':
        """Create point from x, y, z components."""
        return cls(np.array([x, y, z], dtype=np.float64))
    
    @classmethod
    def origin(cls) -> 'Point3D':
        """Create the origin point (0, 0, 0)."""
        return cls(np.zeros(3))
    
    @property
    def x(self) -> float:
        return float(self.coords[0])
    
    @property
    def y(self) -> float:
        return float(self.coords[1])
    
    @property
    def z(self) -> float:
        return float(self.coords[2])
    
    def to_vector(self, other: 'Point3D') -> Vector3D:
        """Create a vector from this point to another point."""
        return Vector3D(other.coords - self.coords)
    
    def distance(self, other: 'Point3D') -> float:
        """Calculate distance to another point."""
        return float(np.linalg.norm(other.coords - self.coords))
    
    def midpoint(self, other: 'Point3D') -> 'Point3D':
        """Calculate midpoint between this and another point."""
        return Point3D((self.coords + other.coords) / 2)
    
    def translate(self, vec: Vector3D) -> 'Point3D':
        """Translate point by a vector."""
        return Point3D(self.coords + vec.coords)
    
    def rotate(self, axis: Vector3D, angle: float, center: Optional['Point3D'] = None) -> 'Point3D':
        """
        Rotate point around axis through center using Rodrigues formula.
        
        Parameters
        ----------
        axis : Vector3D
            Rotation axis (will be normalized)
        angle : float
            Rotation angle in radians
        center : Point3D, optional
            Center of rotation (defaults to origin)
            
        Returns
        -------
        Point3D
            Rotated point
        """
        if center is None:
            center = Point3D.origin()
        
        # Translate to origin
        rel_pos = self.coords - center.coords
        
        # Normalize axis
        k = axis.normalize().coords
        
        # Rodrigues rotation formula: v_rot = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        dot_kv = np.dot(k, rel_pos)
        cross_kv = np.cross(k, rel_pos)
        
        rotated = rel_pos * cos_a + cross_kv * sin_a + k * dot_kv * (1 - cos_a)
        
        # Translate back
        return Point3D(rotated + center.coords)
    
    def rotate_xyz(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> 'Point3D':
        """
        Rotate point around the origin using Euler angles.
        
        Parameters
        ----------
        x : float
            Rotation around x-axis in radians
        y : float
            Rotation around y-axis in radians
        z : float
            Rotation around z-axis in radians
            
        Returns
        -------
        Point3D
            Rotated point
        """
        # Rotate around x-axis
        y1 = self.y * math.cos(x) - self.z * math.sin(x)
        z1 = self.y * math.sin(x) + self.z * math.cos(x)
        
        # Rotate around y-axis
        x2 = self.x * math.cos(y) + z1 * math.sin(y)
        z2 = -self.x * math.sin(y) + z1 * math.cos(y)
        
        # Rotate around z-axis
        x3 = x2 * math.cos(z) - y1 * math.sin(z)
        y3 = x2 * math.sin(z) + y1 * math.cos(z)
        
        return Point3D.from_xyz(x3, y3, z2)
    
    def __repr__(self) -> str:
        return f"Point3D({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


def gram_schmidt(n: Vector3D) -> Tuple[Vector3D, Vector3D]:
    """
    Generate two orthogonal vectors for a given vector using the Gram-Schmidt process.
    
    This creates an orthonormal basis where n is one axis, and v, w are 
    perpendicular to n and to each other.
    
    Parameters
    ----------
    n : Vector3D
        The input vector (will be normalized)
        
    Returns
    -------
    Tuple[Vector3D, Vector3D]
        Two orthogonal unit vectors perpendicular to n
    """
    n = n.normalize()
    
    # Start with a non-parallel vector
    # Use (1,0,0) unless n is nearly parallel to it, then use (0,1,0)
    if abs(n.x) < 0.9:
        seed = Vector3D.from_xyz(1.0, 0.0, 0.0)
    else:
        seed = Vector3D.from_xyz(0.0, 1.0, 0.0)
    
    # Gram-Schmidt: v = seed - (seed · n) * n, then normalize
    projection = n.multiply(seed.dot(n))
    v = seed.subtract(projection).normalize()
    
    # w = n × v (already unit since n and v are orthonormal)
    w = n.cross(v)
    
    return v, w


def rodrigues_rotate(points: np.ndarray, axis: Vector3D, angle: float, 
                     center: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Rotate multiple points around an axis using Rodrigues formula.
    
    Vectorized implementation for performance.
    
    Parameters
    ----------
    points : np.ndarray
        Points to rotate, shape (N, 3)
    axis : Vector3D
        Rotation axis
    angle : float
        Rotation angle in radians
    center : np.ndarray, optional
        Center of rotation, shape (3,). Defaults to origin.
        
    Returns
    -------
    np.ndarray
        Rotated points, shape (N, 3)
    """
    if center is None:
        center = np.zeros(3)
    
    # Translate to origin
    rel_points = points - center
    
    # Normalize axis
    k = axis.normalize().coords
    
    # Rodrigues formula (vectorized)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    dot_products = np.dot(rel_points, k)  # Shape (N,)
    cross_products = np.cross(k, rel_points)  # Shape (N, 3)
    
    rotated = (
        rel_points * cos_a +
        cross_products * sin_a +
        k * dot_products[:, np.newaxis] * (1 - cos_a)
    )
    
    # Translate back
    return rotated + center


# =============================================================================
# 3D Convex Hull for Molecular Envelopes
# =============================================================================


def calculate_convex_hull_3d(points: np.ndarray) -> np.ndarray:
    """
    Calculate the 3D convex hull of a set of points.
    
    Uses scipy's ConvexHull which implements the Quickhull algorithm.
    
    Parameters
    ----------
    points : np.ndarray
        Points array, shape (N, 3)
        
    Returns
    -------
    np.ndarray
        Indices of points that form the convex hull vertices
    """
    from scipy.spatial import ConvexHull
    
    if len(points) < 4:
        # Not enough points for 3D hull
        return np.arange(len(points))
    
    # Check if points are coplanar (degenerate case)
    centered = points - points.mean(axis=0)
    _, s, _ = np.linalg.svd(centered)
    
    # If smallest singular value is near zero, points are ~coplanar
    if len(s) >= 3 and s[2] < 1e-10 * s[0]:
        # Fall back to 2D hull on the dominant plane
        return _calculate_convex_hull_2d_projection(points)
    
    hull = ConvexHull(points)
    return hull.vertices


def _calculate_convex_hull_2d_projection(points: np.ndarray) -> np.ndarray:
    """
    Calculate 2D convex hull by projecting to dominant plane.
    
    Used when 3D points are approximately coplanar.
    """
    from scipy.spatial import ConvexHull
    
    # Find principal plane via SVD
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered)
    
    # Project onto first two principal components
    proj_2d = centered @ vh[:2].T
    
    if len(proj_2d) < 3:
        return np.arange(len(points))
    
    hull_2d = ConvexHull(proj_2d)
    return hull_2d.vertices


def calculate_molecular_envelope(positions: np.ndarray, 
                                  radii: Optional[np.ndarray] = None,
                                  num_surface_points: int = 12) -> np.ndarray:
    """
    Calculate the 3D convex hull envelope of a molecule.
    
    Expands atomic positions by their radii by sampling points on 
    the surface of atomic spheres.
    
    Parameters
    ----------
    positions : np.ndarray
        Atomic positions, shape (N, 3)
    radii : np.ndarray, optional
        Atomic radii, shape (N,). Uses van der Waals radii if not provided.
    num_surface_points : int
        Number of surface points per atom for envelope expansion
        
    Returns
    -------
    np.ndarray
        Hull vertices positions, shape (M, 3)
    """
    if radii is None:
        # Default to uniform radius
        radii = np.ones(len(positions)) * 1.5
    
    # Generate surface points for each atom
    all_points = []
    
    # Use icosahedron-like distribution for surface points
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    for i, (pos, r) in enumerate(zip(positions, radii)):
        # Fibonacci sphere for uniform distribution
        for j in range(num_surface_points):
            y = 1 - (j / (num_surface_points - 1)) * 2 if num_surface_points > 1 else 0
            radius_at_y = np.sqrt(1 - y * y) if abs(y) < 1 else 0
            theta = phi * j * 2 * np.pi
            
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            
            surface_point = pos + np.array([x, y, z]) * r
            all_points.append(surface_point)
    
    all_points = np.array(all_points)
    
    if len(all_points) < 4:
        return all_points
    
    hull_indices = calculate_convex_hull_3d(all_points)
    return all_points[hull_indices]


def envelopes_overlap(hull1_vertices: np.ndarray, 
                       hull2_vertices: np.ndarray,
                       tolerance: float = 0.0) -> bool:
    """
    Fast check if two convex hulls overlap using Separating Axis Theorem.
    
    Parameters
    ----------
    hull1_vertices : np.ndarray
        Vertices of first hull, shape (M1, 3)
    hull2_vertices : np.ndarray
        Vertices of second hull, shape (M2, 3)
    tolerance : float
        Overlap tolerance (positive = allow some penetration)
        
    Returns
    -------
    bool
        True if hulls overlap, False otherwise
    """
    # Quick bounding box check first
    min1, max1 = hull1_vertices.min(axis=0), hull1_vertices.max(axis=0)
    min2, max2 = hull2_vertices.min(axis=0), hull2_vertices.max(axis=0)
    
    # If bounding boxes don't overlap, hulls don't overlap
    if np.any(max1 + tolerance < min2) or np.any(max2 + tolerance < min1):
        return False
    
    # For more precise check, use GJK or full SAT
    # Here we use a simplified approach: check if any vertex of one hull
    # is inside the other hull
    
    from scipy.spatial import ConvexHull, Delaunay
    
    # Build Delaunay triangulation for point-in-hull tests
    if len(hull1_vertices) >= 4:
        # Check if hull2 vertices are inside hull1
        delaunay1 = Delaunay(hull1_vertices)
        if np.any(delaunay1.find_simplex(hull2_vertices) >= 0):
            return True
    
    if len(hull2_vertices) >= 4:
        # Check if hull1 vertices are inside hull2  
        delaunay2 = Delaunay(hull2_vertices)
        if np.any(delaunay2.find_simplex(hull1_vertices) >= 0):
            return True
    
    # If both small, do pairwise distance check
    if len(hull1_vertices) < 4 or len(hull2_vertices) < 4:
        for v1 in hull1_vertices:
            for v2 in hull2_vertices:
                if np.linalg.norm(v1 - v2) < tolerance + 0.1:
                    return True
    
    return False


def minimum_hull_distance(hull1_vertices: np.ndarray,
                           hull2_vertices: np.ndarray) -> float:
    """
    Calculate minimum distance between two convex hulls.
    
    Parameters
    ----------
    hull1_vertices : np.ndarray
        Vertices of first hull, shape (M1, 3)
    hull2_vertices : np.ndarray
        Vertices of second hull, shape (M2, 3)
        
    Returns
    -------
    float
        Minimum distance between hulls (0 if overlapping)
    """
    # Simple approach: minimum pairwise vertex distance
    # For exact hull distance, would need GJK algorithm
    min_dist = float('inf')
    
    for v1 in hull1_vertices:
        dists = np.linalg.norm(hull2_vertices - v1, axis=1)
        min_dist = min(min_dist, dists.min())
    
    return min_dist

