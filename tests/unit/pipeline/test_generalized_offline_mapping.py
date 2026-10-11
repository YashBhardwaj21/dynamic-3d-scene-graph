"""Unit tests for Generalized Offline Semantic Mapping Pipeline components."""

import json
from pathlib import Path
import numpy as np
import pytest

from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.floor_alignment import construct_floor_transform, compute_dynamic_map_bounds
from scene_graph.data.dataset_discovery import discover_recording, load_camera_calibration, RecordingManifest
from scene_graph.pipeline.trajectory_manager import (
    load_tum_trajectory,
    validate_trajectory_alignment,
    find_matching_pose,
    TrajectoryValidationError,
)
from scene_graph.mapping.occupancy_grid import OccupancyGrid, OccupancyGridConfig
from scene_graph.mapping.global_associator import GlobalObjectAssociator, ObjectObservationRecord


def test_floor_transform_orthogonality_and_rep103():
    """Verify construct_floor_transform produces an orthogonal SE(3) matrix with floor at Z=0."""
    # Simulated upward normal tilted by ~10 degrees
    n = np.array([0.1697, -0.9835, -0.0631])
    d = 0.5167

    T_floor_world, T_world_floor = construct_floor_transform(n, d)

    # 1. Check dimensions
    assert T_floor_world.shape == (4, 4)
    assert T_world_floor.shape == (4, 4)

    # 2. Check orthogonality: R @ R.T == I and det(R) == 1
    R = T_floor_world[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6)

    # 3. Inverse check
    assert np.allclose(T_floor_world @ T_world_floor, np.eye(4), atol=1e-6)

    # 4. Point on floor plane: n.x*x + n.y*y + n.z*z + d = 0
    # Choose point p = -d * n
    p_floor_in_world = -d * (n / np.linalg.norm(n))
    p_homo = np.array([p_floor_in_world[0], p_floor_in_world[1], p_floor_in_world[2], 1.0])
    p_in_floor_frame = T_floor_world @ p_homo

    # In floor_map frame, Z coordinate must be identically 0.0!
    assert abs(p_in_floor_frame[2]) < 1e-5


def test_dynamic_map_bounds_adaptation():
    """Verify dynamic bounds scale with trajectory extent and align to resolution multiples."""
    # Trajectory moving from (0, 0) to (5.2, -3.4)
    traj_pts = np.array([
        [0.0, 0.0, 0.5],
        [1.5, -1.0, 0.5],
        [3.0, -2.0, 0.5],
        [5.2, -3.4, 0.5],
    ])

    resolution = 0.05
    padding = 1.0
    ox, oy, w, h = compute_dynamic_map_bounds(
        camera_positions_floor=traj_pts,
        padding_m=padding,
        resolution_m=resolution,
    )

    # Origin should be <= min coords - padding
    assert ox <= 0.0 - padding
    assert oy <= -3.4 - padding

    # Max extent should enclose 5.2 + padding
    assert ox + w >= 5.2 + padding
    assert oy + h >= 0.0 + padding

    # Origin and size must be integer multiples of resolution (within float precision)
    assert abs(round(ox / resolution) - (ox / resolution)) < 1e-4
    assert abs(round(oy / resolution) - (oy / resolution)) < 1e-4
    assert abs(round(w / resolution) - (w / resolution)) < 1e-4
    assert abs(round(h / resolution) - (h / resolution)) < 1e-4


def test_trajectory_validation_rejects_mismatched_timestamps():
    """Verify that validating trajectory with mismatched timestamps raises TrajectoryValidationError."""
    # Fake trajectory from t=100.0 to t=110.0
    fake_poses = {
        100.0: np.eye(4),
        105.0: np.eye(4),
        110.0: np.eye(4),
    }

    # Recording timestamps from t=1791620000.0 (completely different epoch)
    recording_timestamps = [1791620000.0 + i * 0.033 for i in range(100)]

    # Must raise TrajectoryValidationError rather than silently reusing wrong poses!
    with pytest.raises(TrajectoryValidationError) as exc_info:
        validate_trajectory_alignment(fake_poses, recording_timestamps)

    assert "do not overlap" in str(exc_info.value)


def test_trajectory_validation_accepts_aligned_timestamps():
    """Verify that validating trajectory with matching timestamps passes with high match ratio."""
    t0 = 1000.0
    poses = {t0 + i * 0.033: np.eye(4) for i in range(50)}
    rgb_ts = [t0 + i * 0.033 for i in range(50)]

    res = validate_trajectory_alignment(poses, rgb_ts, max_dt=0.05)
    assert res["is_aligned"] is True
    assert res["sample_match_ratio_pct"] >= 90.0


def test_dynamic_discovery_synthetic_recording(tmp_path: Path):
    """Verify dynamic discovery discovers frames, generates DP associations, and loads calibration."""
    seq_dir = tmp_path / "test_seq_001"
    seq_dir.mkdir()
    rgb_dir = seq_dir / "rgb"
    rgb_dir.mkdir()
    depth_dir = seq_dir / "depth"
    depth_dir.mkdir()

    # Create dummy image files
    ts_list = [10.00, 10.033, 10.066, 10.100, 10.133]
    with open(seq_dir / "rgb.txt", "w") as f:
        for t in ts_list:
            p = f"rgb/{t:.3f}.png"
            f.write(f"{t:.3f} {p}\n")
            (seq_dir / p).touch()

    with open(seq_dir / "depth.txt", "w") as f:
        # Slight jitter in depth timestamps (e.g. +4ms)
        for t in ts_list:
            p = f"depth/{t+0.004:.3f}.png"
            f.write(f"{t+0.004:.3f} {p}\n")
            (seq_dir / p).touch()

    # Create custom camera_intrinsics.json
    intr_data = {
        "fx": 400.0,
        "fy": 400.0,
        "cx": 320.0,
        "cy": 240.0,
        "width": 640,
        "height": 480,
        "depth_unit_meters": 0.001,
    }
    with open(seq_dir / "camera_intrinsics.json", "w") as f:
        json.dump(intr_data, f)

    # Discover without precomputed associations.txt
    manifest = discover_recording(seq_dir, max_sync_dt=0.02)

    assert manifest.sequence_name == "test_seq_001"
    assert manifest.total_rgb_count == 5
    assert manifest.total_depth_count == 5
    assert manifest.total_associated_count == 5
    assert manifest.intrinsics.fx == 400.0
    assert manifest.depth_scale_m_per_unit == 0.001
    assert manifest.has_precomputed_associations is False  # Dynamically generated!


def test_occupancy_grid_with_dynamic_bounds_and_inflation():
    """Verify OccupancyGrid initializes with custom bounds and computes inflated stats."""
    cfg = OccupancyGridConfig(
        resolution_m=0.10,
        origin_x_m=-5.0,
        origin_y_m=-4.0,
        width_m=10.0,
        height_m=8.0,
        robot_radius_m=0.20,
        safety_margin_m=0.10,
    )
    grid = OccupancyGrid(cfg)

    assert grid.cols == 100
    assert grid.rows == 80

    # Camera at (0, 0), Obstacle at (2.0, 1.0, 0.5)
    pts = np.array([[2.0, 1.0, 0.5]])
    grid.integrate_depth_points(camera_floor_x=0.0, camera_floor_y=0.0, points_floor_aligned=pts)

    stats = grid.compute_stats()
    assert stats["occupied_cells"] >= 1
    assert stats["free_cells"] >= 1
    assert stats["inflated_occupied_cells"] > stats["occupied_cells"]  # Inflated by 0.30m!
