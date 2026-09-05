"""Reference frame definition and gravity alignment verification."""

from dataclasses import dataclass
import numpy as np


@dataclass
class RelationReferenceFrame:
    """Definition of the stable world coordinate system for relations."""
    name: str                         # e.g., "tum_fr1_desk_gravity_aligned"
    alignment_transform: np.ndarray   # (4,4) matrix
    gravity_axis: str                 # e.g., "z_up"
    horizontal_axes: tuple[str, str]  # e.g., ("x", "y")
    depth_axis: str                   # e.g., "y" (or whichever is front/back)


def estimate_support_plane_normal(points: np.ndarray) -> np.ndarray:
    """Estimate the dominant support plane normal from a point cloud.
    
    Uses Principal Component Analysis (SVD) to find the normal vector.
    Assumes `points` is a dense cloud of the primary desk or floor.
    """
    if len(points) < 3:
        return np.array([0.0, 0.0, 1.0])
        
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # SVD on the centered points
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    
    # The normal is the eigenvector corresponding to the smallest eigenvalue
    normal = vh[2, :]
    
    # Ensure the normal generally points "up" (positive Z)
    if normal[2] < 0:
        normal = -normal
        
    return normal


def compute_alignment_from_normal(normal: np.ndarray) -> np.ndarray:
    """Compute a 4x4 transform that aligns the given normal to [0, 0, 1]."""
    target_up = np.array([0.0, 0.0, 1.0])
    
    normal = normal / np.linalg.norm(normal)
    v = np.cross(normal, target_up)
    c = np.dot(normal, target_up)
    
    if c > 0.999:
        R = np.eye(3)
    elif c < -0.999:
        R = np.diag([1.0, -1.0, -1.0])
    else:
        s = np.linalg.norm(v)
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))
        
    T = np.eye(4)
    T[:3, :3] = R
    return T


def compute_alignment_transform(points_world: np.ndarray) -> np.ndarray:
    """Compute a 4x4 transform that aligns the normal of points_world to [0, 0, 1].
    
    This computes an alignment to the dominant support plane (e.g., desk, floor),
    which is useful for defining a stable world orientation for relations.
    """
    normal = estimate_support_plane_normal(points_world)
    return compute_alignment_from_normal(normal)
