"""Object geometry utilities for point clouds."""

import numpy as np

from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.transforms import transform_points


def compute_object_points_world(
    mask: np.ndarray, 
    depth_m: np.ndarray, 
    intrinsics: CameraIntrinsics, 
    pose: np.ndarray
) -> np.ndarray | None:
    """Compute 3D world points for an object given its 2D mask.
    
    Args:
        mask: 2D boolean or uint8 mask array (H, W).
        depth_m: 2D float metric depth array (H, W) in meters.
        intrinsics: CameraIntrinsics instance.
        pose: 4x4 SE(3) world_T_camera matrix.
        
    Returns:
        (N, 3) float64 array of world points, or None if no valid points.
    """
    if mask.ndim != 2 or depth_m.ndim != 2:
        raise ValueError("Mask and depth must be 2D arrays")
    if mask.shape != depth_m.shape:
        raise ValueError(f"Shape mismatch: mask {mask.shape} != depth {depth_m.shape}")
        
    # Extract valid depth pixels within the mask
    # Ensure depth is positive and finite
    valid_depth_mask = (mask > 0) & (depth_m > 0) & np.isfinite(depth_m)
    v, u = np.where(valid_depth_mask)
    
    if len(u) == 0:
        return None
        
    z_m = depth_m[valid_depth_mask].astype(np.float64)
    
    # Project to camera coordinates
    x = (u - intrinsics.cx) * z_m / intrinsics.fx
    y = (v - intrinsics.cy) * z_m / intrinsics.fy
    
    points_camera = np.stack([x, y, z_m], axis=-1)
    
    # Transform to world coordinates
    points_world = transform_points(pose, points_camera)
    return points_world


def compute_object_robust_center_world(
    mask: np.ndarray, 
    depth_m: np.ndarray, 
    intrinsics: CameraIntrinsics, 
    pose: np.ndarray
) -> np.ndarray | None:
    """Compute a robust 3D center (coordinate-wise median) for an object."""
    points = compute_object_points_world(mask, depth_m, intrinsics, pose)
    if points is None:
        return None
    return np.median(points, axis=0)


def compute_object_aabb_world(
    mask: np.ndarray, 
    depth_m: np.ndarray, 
    intrinsics: CameraIntrinsics, 
    pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute 3D Axis-Aligned Bounding Box (AABB) in world coordinates."""
    points = compute_object_points_world(mask, depth_m, intrinsics, pose)
    if points is None:
        return None
    aabb_min = np.min(points, axis=0)
    aabb_max = np.max(points, axis=0)
    return aabb_min, aabb_max
