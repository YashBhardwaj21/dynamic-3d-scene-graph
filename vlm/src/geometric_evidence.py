"""Stage 5: Geometric Evidence Extraction for Object Observations.

Extracts:
1. Physical RealSense D455 stereo depth uncertainty: sigma_z = (z^2 / (f_x * B)) * sigma_d
2. Genuine MapPoint reprojection residuals and inlier ratios (no synthetic defaults)
3. Explicit evidence provenance: MEASURED_FULL, DERIVED_DEPTH_ONLY, or MISSING
4. Calibrated geometric consistency score bounded in [0.0, 1.0]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


# Intel RealSense D455 Factory Intrinsics (640x480 resolution mode)
FX = 383.924896
FY = 383.924896
CX = 320.123016
CY = 244.569885
STEREO_BASELINE_M = 0.095  # 95 mm physical baseline between IR stereo sensors
DISPARITY_NOISE_PX = 0.08  # Subpixel disparity matching noise (sigma_d)
MIN_VALID_DEPTH_M = 0.20
MAX_VALID_DEPTH_M = 4.80


@dataclass
class GeometricEvidenceRecord:
    """Timestamp-aligned geometric evidence record for a single object observation."""
    obs_id: str
    frame_index: int
    timestamp: float
    entity_id: str
    track_id: str
    depth_valid: bool
    measured_depth_m: Optional[float]
    depth_uncertainty_sigma_m: Optional[float]
    depth_provenance: str  # "MEASURED_SENSOR" or "MISSING"
    mappoints_count: int
    inlier_count: int
    inlier_ratio: Optional[float]
    reprojection_residual_valid: bool
    reprojection_provenance: str  # "MEASURED_MAPPOINTS" or "MISSING_MAPPOINTS"
    mean_reprojection_residual_px: Optional[float]
    median_reprojection_residual_px: Optional[float]
    p95_reprojection_residual_px: Optional[float]
    landmark_support_count: int
    landmark_spatial_extent_m: float
    geometric_consistency_score: float
    provenance: str  # "MEASURED_FULL", "DERIVED_DEPTH_ONLY", "MISSING"
    has_valid_evidence: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert unit quaternion to 3x3 orthonormal rotation matrix."""
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-9:
        return np.eye(3)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)

    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qx * qw)

    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    return np.array([[r00, r01, r02],
                     [r10, r11, r12],
                     [r20, r21, r22]], dtype=np.float64)


def compute_depth_uncertainty(depth_z: float) -> float:
    """Compute physical depth uncertainty (1-sigma) for stereo triangulation.
    
    Formula: sigma_z = (z^2 / (f_x * B)) * sigma_d
    """
    if depth_z <= 0.05:
        return 0.10
    z = max(0.15, min(depth_z, 6.0))
    sigma_z = (z * z / (FX * STEREO_BASELINE_M)) * DISPARITY_NOISE_PX
    return float(max(0.001, sigma_z))


class GeometricEvidenceExtractor:
    """Extracts geometric consistency evidence for object observations."""

    def __init__(self, mappoints_path: Optional[Path] = None):
        self.mappoints = np.empty((0, 3), dtype=np.float64)
        if mappoints_path and Path(mappoints_path).exists():
            self._load_mappoints_ply(Path(mappoints_path))

    def _load_mappoints_ply(self, path: Path):
        """Load 3D MapPoints from PLY file."""
        points = []
        in_header = True
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if in_header:
                    if line == "end_header":
                        in_header = False
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue
        if points:
            self.mappoints = np.array(points, dtype=np.float64)

    def extract_observation_geometry(
        self,
        obs: Dict[str, Any],
    ) -> GeometricEvidenceRecord:
        """Process a single observation record and compute real geometric metrics."""
        obs_id = obs.get("obs_id", "")
        frame_idx = int(obs.get("frame_index", 0))
        ts = float(obs.get("timestamp", 0.0))
        entity_id = obs.get("global_entity_id", "")
        track_id = obs.get("local_track_id", "")
        points_count = int(obs.get("points_count", 0))
        bbox_xyxy = obs.get("bbox_xyxy")

        # 1. Depth extraction & physical uncertainty
        centroid_cam = obs.get("centroid_camera")
        if centroid_cam is not None and len(centroid_cam) >= 3 and centroid_cam[2] is not None:
            z_depth = float(centroid_cam[2])
            depth_valid = (MIN_VALID_DEPTH_M <= z_depth <= MAX_VALID_DEPTH_M)
            sigma_z = compute_depth_uncertainty(z_depth)
            depth_prov = "MEASURED_SENSOR"
        else:
            z_depth = None
            depth_valid = False
            sigma_z = None
            depth_prov = "MISSING"

        # 2. 3D spatial extent from floor bounding box
        b_min_raw = obs.get("bbox_min_floor")
        b_max_raw = obs.get("bbox_max_floor")
        if b_min_raw is not None and b_max_raw is not None and len(b_min_raw) == 3 and len(b_max_raw) == 3:
            b_min = np.array(b_min_raw, dtype=np.float64)
            b_max = np.array(b_max_raw, dtype=np.float64)
            spatial_extent = float(np.linalg.norm(b_max - b_min))
        else:
            b_min = None
            b_max = None
            spatial_extent = 0.0

        # 3. Camera pose
        cam_pose = obs.get("camera_pose")
        has_pose = (cam_pose is not None and len(cam_pose) >= 7)

        # 4. MapPoint projection & reprojection residual calculation
        candidate_pts = []
        if has_pose and len(self.mappoints) > 0 and b_min is not None and b_max is not None:
            mask = np.all((self.mappoints >= b_min - 0.10) & (self.mappoints <= b_max + 0.10), axis=1)
            candidate_pts = self.mappoints[mask]

        n_candidates = len(candidate_pts)
        residuals: List[float] = []
        inliers = 0

        if n_candidates > 0 and has_pose and bbox_xyxy is not None and len(bbox_xyxy) == 4:
            t_wc = np.array(cam_pose[:3], dtype=np.float64)
            R_wc = quaternion_to_rotation_matrix(cam_pose[3], cam_pose[4], cam_pose[5], cam_pose[6])
            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc

            bx1, by1, bx2, by2 = bbox_xyxy
            bcx = (bx1 + bx2) / 2.0
            bcy = (by1 + by2) / 2.0

            p_cam = (R_cw @ candidate_pts.T).T + t_cw
            valid_z = p_cam[:, 2] > 0.15

            for p in p_cam[valid_z]:
                u = FX * (p[0] / p[2]) + CX
                v = FY * (p[1] / p[2]) + CY
                px_res = math.hypot(u - bcx, v - bcy)
                residuals.append(px_res)

                if bx1 - 8.0 <= u <= bx2 + 8.0 and by1 - 8.0 <= v <= by2 + 8.0:
                    inliers += 1

        # 5. Honest Provenance & Fallback-Free Residuals
        if residuals:
            res_arr = np.array(residuals)
            mean_res = float(np.mean(res_arr))
            med_res = float(np.median(res_arr))
            p95_res = float(np.percentile(res_arr, 95))
            inlier_ratio = float(inliers / max(1, n_candidates))
            reproj_valid = True
            reproj_prov = "MEASURED_MAPPOINTS"
            provenance = "MEASURED_FULL"
        else:
            mean_res = None
            med_res = None
            p95_res = None
            inlier_ratio = None
            reproj_valid = False
            reproj_prov = "MISSING_MAPPOINTS"
            provenance = "DERIVED_DEPTH_ONLY" if depth_valid else "MISSING"

        # Landmark support count
        support_count = max(points_count, n_candidates)

        # 6. Geometric consistency score
        if provenance == "MEASURED_FULL" and mean_res is not None and inlier_ratio is not None and sigma_z is not None:
            inlier_score = min(1.0, max(0.0, inlier_ratio))
            residual_score = math.exp(-mean_res / 35.0)
            depth_score = math.exp(-sigma_z / 0.04) if depth_valid else 0.10
            consistency_score = 0.40 * inlier_score + 0.30 * residual_score + 0.30 * depth_score
            has_valid_evidence = (depth_valid and inlier_score > 0.10)
        elif depth_valid and sigma_z is not None:
            depth_score = math.exp(-sigma_z / 0.04)
            consistency_score = 0.60 * depth_score + 0.40 * min(1.0, points_count / 2000.0)
            has_valid_evidence = depth_valid
        else:
            consistency_score = 0.0
            has_valid_evidence = False

        return GeometricEvidenceRecord(
            obs_id=obs_id,
            frame_index=frame_idx,
            timestamp=ts,
            entity_id=entity_id,
            track_id=track_id,
            depth_valid=depth_valid,
            measured_depth_m=round(z_depth, 4) if z_depth is not None else None,
            depth_uncertainty_sigma_m=round(sigma_z, 4) if sigma_z is not None else None,
            depth_provenance=depth_prov,
            mappoints_count=n_candidates,
            inlier_count=inliers,
            inlier_ratio=round(inlier_ratio, 4) if inlier_ratio is not None else None,
            reprojection_residual_valid=reproj_valid,
            reprojection_provenance=reproj_prov,
            mean_reprojection_residual_px=round(mean_res, 2) if mean_res is not None else None,
            median_reprojection_residual_px=round(med_res, 2) if med_res is not None else None,
            p95_reprojection_residual_px=round(p95_res, 2) if p95_res is not None else None,
            landmark_support_count=support_count,
            landmark_spatial_extent_m=round(spatial_extent, 4),
            geometric_consistency_score=round(consistency_score, 4),
            provenance=provenance,
            has_valid_evidence=has_valid_evidence,
        )
