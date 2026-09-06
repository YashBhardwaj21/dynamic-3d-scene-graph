"""Object geometry utilities for point clouds."""

import numpy as np

from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.transforms import transform_points


from dataclasses import dataclass
from typing import Optional, Tuple, Dict
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
    points_world_sampled: Optional[np.ndarray] = None
    centroid_camera: Optional[np.ndarray] = None
    centroid_world: Optional[np.ndarray] = None
    position_covariance_world: Optional[np.ndarray] = None
    point_covariance_world: Optional[np.ndarray] = None
    bbox_min_world: Optional[np.ndarray] = None
    bbox_max_world: Optional[np.ndarray] = None
    obb_center_world: Optional[np.ndarray] = None
    obb_axes_world: Optional[np.ndarray] = None
    obb_extents_world: Optional[np.ndarray] = None
    geometry_quality: float = 0.0
    depth_stats: Optional[Dict[str, float]] = None
    valid_point_count: int = 0
    status: GeometryStatus = GeometryStatus.VALID


def compute_object_geometry(
    mask: np.ndarray, 
    depth_m: np.ndarray, 
    intrinsics: CameraIntrinsics, 
    pose: np.ndarray,
    min_valid_points: int = 30,
    mad_k: float = 3.5,
    voxel_size_m: float = 0.005
) -> ObjectGeometry | None:
    """Compute robust 3D geometry for an object, mitigating background leakage.
    
    Args:
        mask: 2D boolean mask.
        depth_m: 2D depth in meters.
        intrinsics: Camera parameters.
        pose: 4x4 SE(3) world_T_camera matrix.
        min_valid_points: Minimum number of valid points required.
        mad_k: Number of robust sigmas for MAD outlier rejection.
        voxel_size_m: Voxel size for downsampling.
        
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
    
    # Depth-robust extraction policy (MAD)
    median = np.median(z_m)
    mad = np.median(np.abs(z_m - median))
    sigma_robust = 1.4826 * mad
    if sigma_robust < 1e-6:
        sigma_robust = 1e-6
    band_mask = np.abs(z_m - median) <= mad_k * sigma_robust
    
    if np.sum(band_mask) < min_valid_points:
        return ObjectGeometry(status=GeometryStatus.INSUFFICIENT_DEPTH)
        
    u_filt = u[band_mask]
    v_filt = v[band_mask]
    z_filt = z_m[band_mask]
    
    # Project to camera coordinates using vectorized projection
    points_camera = intrinsics.pixels_to_camera(u_filt, v_filt, z_filt)
    
    # Transform to world coordinates
    points_world = transform_points(pose, points_camera)
    
    # Compute centers (mean, rigid-transform equivariant)
    N = len(points_world)
    center_camera = np.mean(points_camera, axis=0)
    center_world = np.mean(points_world, axis=0)
    
    # Covariance
    centered = points_world - center_world
    if N > 1:
        Sigma_P = np.dot(centered.T, centered) / (N - 1)
    else:
        Sigma_P = np.zeros((3, 3))
        
    Sigma_sensor = np.eye(3) * (0.01 ** 2)
    Sigma_c = Sigma_P / N + Sigma_sensor
    
    # Percentile AABB
    aabb_min = np.percentile(points_world, 2, axis=0)
    aabb_max = np.percentile(points_world, 98, axis=0)
    
    # OBB via PCA
    if N > 2:
        cov = np.dot(centered.T, centered) / N
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        obb_axes = eigenvectors[:, idx]
        projected = np.dot(centered, obb_axes)
        min_proj = np.min(projected, axis=0)
        max_proj = np.max(projected, axis=0)
        obb_extents = max_proj - min_proj
        obb_center = center_world + np.dot(obb_axes, (max_proj + min_proj) / 2)
    else:
        obb_axes = np.eye(3)
        obb_extents = np.zeros(3)
        obb_center = center_world
    
    # Compute depth statistics
    depth_stats = {
        "median": float(np.median(z_filt)),
        "p05": float(np.percentile(z_filt, 5)),
        "p25": float(np.percentile(z_filt, 25)),
        "p75": float(np.percentile(z_filt, 75)),
        "p95": float(np.percentile(z_filt, 95))
    }
    
    # Deterministic voxel downsampling
    from scene_graph.geometry.downsampling import voxel_downsample
    points_world_sampled = voxel_downsample(points_world, voxel_size_m)
    
    # Quality heuristic
    geometry_quality = 1.0 if N > min_valid_points * 2 else float(N) / (min_valid_points * 2)
    
    return ObjectGeometry(
        points_camera=points_camera,
        points_world=points_world,
        points_world_sampled=points_world_sampled,
        centroid_camera=center_camera,
        centroid_world=center_world,
        position_covariance_world=Sigma_c,
        point_covariance_world=Sigma_P,
        bbox_min_world=aabb_min,
        bbox_max_world=aabb_max,
        obb_center_world=obb_center,
        obb_axes_world=obb_axes,
        obb_extents_world=obb_extents,
        geometry_quality=geometry_quality,
        depth_stats=depth_stats,
        valid_point_count=len(points_world),
        status=GeometryStatus.VALID
    )
