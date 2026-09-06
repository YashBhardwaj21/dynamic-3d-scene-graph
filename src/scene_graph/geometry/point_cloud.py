"""Object geometry utilities for point clouds."""

import numpy as np

from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.transforms import transform_points


from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

class GeometryStatus(str, Enum):
    VALID = "VALID"
    INSUFFICIENT_DEPTH = "INSUFFICIENT_DEPTH"
    INVALID_GEOMETRY = "INVALID_GEOMETRY"
    NO_DEPTH = "NO_DEPTH"
    NO_POSE = "NO_POSE"

@dataclass
class ObjectGeometry:
    """Cached per-observation geometry."""
    points_camera: Optional[np.ndarray] = None
    points_world: Optional[np.ndarray] = None
    centroid_camera: Optional[np.ndarray] = None
    centroid_world: Optional[np.ndarray] = None
    bbox_min_world: Optional[np.ndarray] = None
    bbox_max_world: Optional[np.ndarray] = None
    valid_point_count: int = 0
    status: GeometryStatus = GeometryStatus.VALID


def compute_object_geometry(
    mask: np.ndarray, 
    depth_m: np.ndarray, 
    intrinsics: CameraIntrinsics, 
    pose: np.ndarray,
    min_valid_points: int = 30,
    depth_outlier_band_m: float = 0.10
) -> ObjectGeometry | None:
    """Compute robust 3D geometry for an object, mitigating background leakage.
    
    Args:
        mask: 2D boolean mask.
        depth_m: 2D depth in meters.
        intrinsics: Camera parameters.
        pose: 4x4 SE(3) world_T_camera matrix.
        min_valid_points: Minimum number of valid points required.
        depth_outlier_band_m: Margin around median depth to retain.
        
    Returns:
        ObjectGeometry (or None if totally invalid input)
    """
    if mask.ndim != 2 or depth_m.ndim != 2:
        raise ValueError("Mask and depth must be 2D arrays")
    if mask.shape != depth_m.shape:
        raise ValueError(f"Shape mismatch: mask {mask.shape} != depth {depth_m.shape}")
        
    # Extract valid depth pixels within the mask
    valid_depth_mask = (mask > 0) & (depth_m > 0) & np.isfinite(depth_m)
    v, u = np.where(valid_depth_mask)
    
    if len(u) < min_valid_points:
        return ObjectGeometry(status=GeometryStatus.INSUFFICIENT_DEPTH)
        
    z_m = depth_m[valid_depth_mask].astype(np.float64)
    
    # Depth-robust extraction policy
    z_med = np.median(z_m)
    band_mask = np.abs(z_m - z_med) <= depth_outlier_band_m
    
    if np.sum(band_mask) < min_valid_points:
        return ObjectGeometry(status=GeometryStatus.INSUFFICIENT_DEPTH)
        
    u_filt = u[band_mask]
    v_filt = v[band_mask]
    z_filt = z_m[band_mask]
    
    # Project to camera coordinates
    x = (u_filt - intrinsics.cx) * z_filt / intrinsics.fx
    y = (v_filt - intrinsics.cy) * z_filt / intrinsics.fy
    
    points_camera = np.stack([x, y, z_filt], axis=-1)
    
    # Transform to world coordinates
    points_world = transform_points(pose, points_camera)
    
    # Compute centers and AABB
    center_camera = np.median(points_camera, axis=0)
    center_world = np.median(points_world, axis=0)
    aabb_min = np.min(points_world, axis=0)
    aabb_max = np.max(points_world, axis=0)
    
    return ObjectGeometry(
        points_camera=points_camera,
        points_world=points_world,
        centroid_camera=center_camera,
        centroid_world=center_world,
        bbox_min_world=aabb_min,
        bbox_max_world=aabb_max,
        valid_point_count=len(points_world),
        status=GeometryStatus.VALID
    )
