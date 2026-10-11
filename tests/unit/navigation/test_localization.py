"""Unit tests for Persistent Map Localization Engine."""

import math
import numpy as np
import pytest
import cv2

from scene_graph.navigation.localization import PersistentMapLocalizer, LocalizationEstimate
from scene_graph.geometry.coordinate_frames import RobotExtrinsics


@pytest.fixture
def temp_map_dir(tmp_path):
    # Create minimal valid floor_plane.json
    fp_data = {
        "frame_id": "floor_map",
        "parent_frame_id": "slam_world",
        "upward_normal": [0.0, -1.0, 0.0],
        "plane_distance_d": 1.0,
        "T_floor_world": [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        "T_world_floor": [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    }
    import json
    with open(tmp_path / "floor_plane.json", "w") as f:
        json.dump(fp_data, f)
    return tmp_path


def test_localizer_initialization_and_transforms(temp_map_dir):
    ext = RobotExtrinsics(x_offset_m=0.12, y_offset_m=0.0, z_offset_m=0.16)
    localizer = PersistentMapLocalizer(temp_map_dir, extrinsics=ext)

    # Test processing a synthetic image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some random high-contrast corners to trigger ORB
    for _ in range(50):
        x, y = np.random.randint(50, 590), np.random.randint(50, 430)
        cv2.rectangle(img, (x, y), (x + 20, y + 20), (255, 255, 255), -1)

    prior_pose = np.eye(4, dtype=np.float64)
    est = localizer.process_frame(img, prior_world_pose=prior_pose)

    assert est.is_valid is True
    assert est.status in ("TRACKED", "DEGRADED")
    assert est.inlier_feature_count > 0
    assert len(est.robot_pose_2d) == 3

    # Transforms must satisfy SE(3) quality gates
    assert est.T_floor_camera.shape == (4, 4)
    assert est.T_floor_base.shape == (4, 4)
