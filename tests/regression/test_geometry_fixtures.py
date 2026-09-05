"""Regression tests against static geometry fixtures (Substage 1.6)."""

import json
from pathlib import Path
import numpy as np
import pytest

from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.transforms import pose_to_transform, transform_points


WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = WORKSPACE_DIR / "data" / "fixtures"
SAMPLES_PATH = FIXTURES_DIR / "geometry_samples.json"


@pytest.mark.skipif(not SAMPLES_PATH.is_file(), reason="Geometry samples fixture missing")
def test_geometry_samples_regression():
    """Verify point projection and transformation match manually calculated or known ground truths."""
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    cam_data = data["camera"]
    intrinsics = CameraIntrinsics(
        fx=cam_data["fx"],
        fy=cam_data["fy"],
        cx=cam_data["cx"],
        cy=cam_data["cy"]
    )
    
    for i, sample in enumerate(data["samples"]):
        # 1. Project depth pixel to camera space
        depth_m = sample["depth_val"] / 5000.0
        x_c, y_c, z_c = intrinsics.pixel_to_camera(sample["u"], sample["v"], depth_m)
        points_camera = np.array([[x_c, y_c, z_c]])
        
        # 2. Build pose transform
        T = pose_to_transform(
            sample["tx"], sample["ty"], sample["tz"],
            sample["qx"], sample["qy"], sample["qz"], sample["qw"]
        )
        
        # 3. Transform to world
        points_world = transform_points(T, points_camera)
        
        # 4. Assert correctness
        expected = np.array([[
            sample["expected_world_x"],
            sample["expected_world_y"],
            sample["expected_world_z"]
        ]])
        np.testing.assert_allclose(
            points_world, 
            expected, 
            err_msg=f"Sample {i} failed regression check"
        )
