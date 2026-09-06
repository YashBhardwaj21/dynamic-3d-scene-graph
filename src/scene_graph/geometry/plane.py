import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class PlaneGeometry:
    normal: np.ndarray        # (3,)
    distance: float           # scalar d in a*x + b*y + c*z + d = 0
    inlier_mask: np.ndarray   # boolean mask of shape (N,)
    residual_mean: float
    residual_std: float

def fit_plane_ransac(points: np.ndarray, distance_threshold: float, max_iterations: int = 100, min_inliers: int = 3) -> Optional[PlaneGeometry]:
    """Fit a plane to 3D points using RANSAC.
    
    Args:
        points: (N, 3) numpy array of 3D points.
        distance_threshold: Maximum distance to the plane for a point to be an inlier.
        max_iterations: Number of RANSAC iterations.
        min_inliers: Minimum number of inliers required to consider a fit valid.
        
    Returns:
        PlaneGeometry if a valid plane was found, else None.
    """
    if len(points) < 3:
        return None
        
    best_plane = None
    best_inliers_count = 0
    best_inlier_mask = None
    
    for _ in range(max_iterations):
        # Sample 3 random points
        indices = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[indices]
        
        # Calculate normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
            
        normal = normal / norm
        distance = -np.dot(normal, p1)
        
        # Calculate distances of all points to the plane
        distances = np.abs(np.dot(points, normal) + distance)
        
        # Find inliers
        inlier_mask = distances <= distance_threshold
        inliers_count = np.sum(inlier_mask)
        
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_plane = (normal, distance)
            best_inlier_mask = inlier_mask
            
    if best_inliers_count < min_inliers or best_plane is None:
        return None
        
    # Refine with all inliers using PCA/SVD
    inlier_points = points[best_inlier_mask]
    centroid = np.mean(inlier_points, axis=0)
    centered = inlier_points - centroid
    cov = np.dot(centered.T, centered) / len(inlier_points)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # The normal is the eigenvector corresponding to the smallest eigenvalue
    normal = eigenvectors[:, 0]
    # Ensure normal points consistently (e.g. positive z)
    if normal[2] < 0:
        normal = -normal
        
    distance = -np.dot(normal, centroid)
    
    # Recompute residuals and mask for the refined plane
    final_distances = np.abs(np.dot(points, normal) + distance)
    final_inlier_mask = final_distances <= distance_threshold
    
    # Calculate residuals of inliers only
    inlier_residuals = final_distances[final_inlier_mask]
    
    return PlaneGeometry(
        normal=normal,
        distance=distance,
        inlier_mask=final_inlier_mask,
        residual_mean=float(np.mean(inlier_residuals)) if len(inlier_residuals) > 0 else 0.0,
        residual_std=float(np.std(inlier_residuals)) if len(inlier_residuals) > 0 else 0.0
    )
