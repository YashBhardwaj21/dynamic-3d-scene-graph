"""SE(3) transform utilities."""

import numpy as np


def quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a quaternion (TUM convention) to a 3x3 rotation matrix.
    
    Args:
        qx, qy, qz, qw: Quaternion components.
        
    Returns:
        A 3x3 numpy array representing the rotation matrix.
    """
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Quaternion norm must be non-zero")
    q = q / norm
    qx, qy, qz, qw = q
    
    R = np.array([
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ], dtype=np.float64)
    
    return R


def pose_to_transform(tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert translation and quaternion to 4x4 SE(3) transformation matrix (world_T_camera).
    
    Args:
        tx, ty, tz: Translation components.
        qx, qy, qz, qw: Quaternion components.
        
    Returns:
        A 4x4 numpy array representing the SE(3) transform.
    """
    R = quaternion_to_matrix(qx, qy, qz, qw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def transform_points(T: np.ndarray, points_camera: np.ndarray) -> np.ndarray:
    """Transform an Nx3 array of 3D points from camera to world coordinates.
    
    Args:
        T: 4x4 SE(3) transformation matrix (world_T_camera).
        points_camera: Nx3 numpy array of points in camera coordinates.
        
    Returns:
        Nx3 numpy array of points in world coordinates.
    """
    if T.shape != (4, 4):
        raise ValueError(f"Transform matrix must be 4x4, got {T.shape}")
    if points_camera.ndim != 2 or points_camera.shape[1] != 3:
        raise ValueError(f"Points must be Nx3, got {points_camera.shape}")
        
    if len(points_camera) == 0:
        return np.empty((0, 3), dtype=np.float64)
        
    # N x 4 homogeneous coordinates
    points_h = np.hstack([points_camera, np.ones((len(points_camera), 1), dtype=np.float64)])
    
    # Transform: (T @ points_h.T).T => points_h @ T.T
    points_world_h = points_h @ T.T
    
    # Return Nx3
    return points_world_h[:, :3]
