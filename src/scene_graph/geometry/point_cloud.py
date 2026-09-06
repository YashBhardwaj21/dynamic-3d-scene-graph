"""Object geometry utilities for point clouds."""

import numpy as np

from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.transforms import transform_points


from dataclasses import dataclass
from typing import Optional

@dataclass
class ObjectGeometry:
    """Cached per-observation geometry."""
    points_camera: np.ndarray        # (N, 3)
    points_world: np.ndarray         # (N, 3)
    robust_center_camera: np.ndarray # (3,)
    robust_center_world: np.ndarray  # (3,)
    aabb_min_world: np.ndarray       # (3,)
    aabb_max_world: np.ndarray       # (3,)
    valid_point_count: int


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
        ObjectGeometry or None if invalid.
    """
    if mask.ndim != 2 or depth_m.ndim != 2:
        raise ValueError("Mask and depth must be 2D arrays")
    if mask.shape != depth_m.shape:
        raise ValueError(f"Shape mismatch: mask {mask.shape} != depth {depth_m.shape}")
        
    # Extract valid depth pixels within the mask
    valid_depth_mask = (mask > 0) & (depth_m > 0) & np.isfinite(depth_m)
    v, u = np.where(valid_depth_mask)
    
    if len(u) < min_valid_points:
        return None
        
    z_m = depth_m[valid_depth_mask].astype(np.float64)
    
    # Depth-robust extraction policy
    z_med = np.median(z_m)
    band_mask = np.abs(z_m - z_med) <= depth_outlier_band_m
    
    if np.sum(band_mask) < min_valid_points:
        return None
        
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
        robust_center_camera=center_camera,
        robust_center_world=center_world,
        aabb_min_world=aabb_min,
        aabb_max_world=aabb_max,
        valid_point_count=len(points_world)
    )
