"""Object geometry utilities for point clouds."""

import numpy as np

from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.transforms import transform_points


def compute_object_points_world(
    mask: np.ndarray, 
    depth: np.ndarray, 
    intrinsics: CameraIntrinsics, 
    pose: np.ndarray
) -> np.ndarray | None:
    """Compute 3D world points for an object given its 2D mask.
    
    Args:
        mask: 2D boolean or uint8 mask array (H, W).
        depth: 2D uint16 depth array (H, W).
        intrinsics: CameraIntrinsics instance.
        pose: 4x4 SE(3) world_T_camera matrix.
        
    Returns:
        (N, 3) float64 array of world points, or None if no valid points.
    """
    # Extract valid depth pixels within the mask
    # TUM depth scale is 5000.0 (raw uint16 / 5000 = meters)
    valid_depth_mask = (mask > 0) & (depth > 0)
    v, u = np.where(valid_depth_mask)
    
    if len(u) == 0:
        return None
        
    z_m = depth[valid_depth_mask].astype(np.float64) / 5000.0
    
    # Project to camera coordinates
    x = (u - intrinsics.cx) * z_m / intrinsics.fx
    y = (v - intrinsics.cy) * z_m / intrinsics.fy
    
    points_camera = np.stack([x, y, z_m], axis=-1)
    
    # Transform to world coordinates
    points_world = transform_points(pose, points_camera)
    return points_world


def compute_object_centroid_world(
    mask: np.ndarray, 
    depth: np.ndarray, 
    intrinsics: CameraIntrinsics, 
    pose: np.ndarray
) -> np.ndarray | None:
    """Compute median 3D world centroid for an object."""
    points = compute_object_points_world(mask, depth, intrinsics, pose)
    if points is None:
        return None
    return np.median(points, axis=0)


def compute_object_bbox_world(
    mask: np.ndarray, 
    depth: np.ndarray, 
    intrinsics: CameraIntrinsics, 
    pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute 3D world bounding box for an object."""
    points = compute_object_points_world(mask, depth, intrinsics, pose)
    if points is None:
        return None
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    return bbox_min, bbox_max
