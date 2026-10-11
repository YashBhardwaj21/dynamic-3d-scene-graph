"""Trajectory Ingestion, Validation, and Synchronization with Recording Frames."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from scene_graph.geometry.transforms import pose_to_transform


class TrajectoryValidationError(ValueError):
    """Raised when trajectory fails provenance or timestamp validation."""
    pass


def load_tum_trajectory(traj_path: str | Path) -> Dict[float, np.ndarray]:
    """Load TUM trajectory format: timestamp tx ty tz qx qy qz qw -> 4x4 SE(3) matrix."""
    poses: Dict[float, np.ndarray] = {}
    path = Path(traj_path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [float(p) for p in line.split()]
            if len(parts) >= 8:
                t = parts[0]
                tx, ty, tz = parts[1:4]
                qx, qy, qz, qw = parts[4:8]
                T_wc = pose_to_transform(tx, ty, tz, qx, qy, qz, qw)
                poses[t] = T_wc

    if not poses:
        raise ValueError(f"No valid TUM poses found in {traj_path}")
    return poses


def validate_trajectory_alignment(
    poses: Dict[float, np.ndarray],
    rgb_timestamps: List[float],
    max_dt: float = 0.05,
    min_match_ratio: float = 0.20,
) -> Dict[str, Any]:
    """Validate that trajectory timestamps correspond to the recording's actual timestamps.
    
    Prevents silent reuse of trajectories from a different recording.
    """
    if not poses:
        raise TrajectoryValidationError("Trajectory poses dictionary is empty!")
    if not rgb_timestamps:
        raise TrajectoryValidationError("RGB timestamps list is empty!")

    pose_keys = np.array(sorted(poses.keys()))
    t_min_traj, t_max_traj = float(pose_keys[0]), float(pose_keys[-1])
    t_min_rgb, t_max_rgb = float(rgb_timestamps[0]), float(rgb_timestamps[-1])

    overlap_start = max(t_min_traj, t_min_rgb)
    overlap_end = min(t_max_traj, t_max_rgb)
    overlap_duration_s = max(0.0, overlap_end - overlap_start)
    rgb_duration_s = max(0.001, t_max_rgb - t_min_rgb)
    traj_duration_s = max(0.001, t_max_traj - t_min_traj)

    # Check for complete disconnect
    if overlap_duration_s <= 0.0:
        raise TrajectoryValidationError(
            f"Trajectory timestamps [{t_min_traj:.3f}, {t_max_traj:.3f}] do not overlap "
            f"with recording RGB timestamps [{t_min_rgb:.3f}, {t_max_rgb:.3f}]. "
            "Never silently reuse a trajectory from a different sequence!"
        )

    # Match test on sampled points
    matched_count = 0
    test_sample = rgb_timestamps[::max(1, len(rgb_timestamps) // 50)]
    for t in test_sample:
        idx = np.searchsorted(pose_keys, t)
        candidates = []
        if idx < len(pose_keys):
            candidates.append(pose_keys[idx])
        if idx > 0:
            candidates.append(pose_keys[idx - 1])
        if candidates and min(abs(c - t) for c in candidates) <= max_dt:
            matched_count += 1

    match_ratio = matched_count / len(test_sample) if test_sample else 0.0
    if match_ratio < min_match_ratio:
        raise TrajectoryValidationError(
            f"Trajectory matching ratio {match_ratio*100:.1f}% is below acceptable threshold "
            f"{min_match_ratio*100:.1f}% (max_dt={max_dt}s). The trajectory does not align "
            f"with the recording's frames!"
        )

    return {
        "trajectory_poses_count": len(poses),
        "trajectory_start_t": t_min_traj,
        "trajectory_end_t": t_max_traj,
        "trajectory_duration_s": round(traj_duration_s, 2),
        "overlap_duration_s": round(overlap_duration_s, 2),
        "temporal_overlap_pct": round((overlap_duration_s / rgb_duration_s) * 100.0, 1),
        "sample_match_ratio_pct": round(match_ratio * 100.0, 1),
        "is_aligned": True,
    }


def find_matching_pose(
    timestamp: float,
    poses: Dict[float, np.ndarray],
    pose_keys: np.ndarray,
    tolerance_s: float = 0.05,
) -> Optional[np.ndarray]:
    """Find the closest pose within tolerance_s via binary search."""
    idx = np.searchsorted(pose_keys, timestamp)
    candidates = []
    if idx < len(pose_keys):
        candidates.append(pose_keys[idx])
    if idx > 0:
        candidates.append(pose_keys[idx - 1])
    if not candidates:
        return None
    best_t = min(candidates, key=lambda t: abs(t - timestamp))
    if abs(best_t - timestamp) <= tolerance_s:
        return poses[best_t]
    return None
