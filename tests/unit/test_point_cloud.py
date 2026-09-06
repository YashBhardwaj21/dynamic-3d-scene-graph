"""Unit tests for point cloud utilities (Substage 1.8)."""

import numpy as np
import pytest

from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.geometry.point_cloud import compute_object_geometry
from scene_graph.geometry.transforms import pose_to_transform


@pytest.fixture
def dummy_scene():
    """Setup a small synthetic mask, depth map, and pose for testing."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    depth_raw = np.zeros((10, 10), dtype=np.uint16)
    
    # Create a 2x2 object in the mask
    mask[4:6, 4:6] = 255
    # Set depth values to 5000 (1.0 meter)
    depth_raw[4:6, 4:6] = 5000
    
    depth_model = DepthModel(scale=5000.0)
    depth_m = depth_model.depth_to_meters(depth_raw)
    
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=5.0, cy=5.0, width=10, height=10)
    
    # Identity pose
    pose = pose_to_transform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    
    return mask, depth_m, intrinsics, pose


def test_compute_object_geometry(dummy_scene):
    mask, depth_m, intrinsics, pose = dummy_scene
    
    geo = compute_object_geometry(mask, depth_m, intrinsics, pose, min_valid_points=1)
    
    assert geo is not None
    assert geo.points_world.shape == (4, 3)
    
    # Expected: z = 1.0
    np.testing.assert_array_equal(geo.points_world[:, 2], 1.0)
    
    # u, v range is [4, 5] x [4, 5]
    # x = (u - cx) * z / fx = (4 - 5) / 500 = -0.002
    # x = (5 - 5) / 500 = 0.0
    expected_x_set = set([-0.002, 0.0])
    expected_y_set = set([-0.002, 0.0])
    
    actual_x_set = set(geo.points_world[:, 0])
    actual_y_set = set(geo.points_world[:, 1])
    
    assert actual_x_set == expected_x_set
    assert actual_y_set == expected_y_set
    
    assert geo.robust_center_world[0] == pytest.approx(-0.001)
    assert geo.robust_center_world[1] == pytest.approx(-0.001)
    assert geo.robust_center_world[2] == pytest.approx(1.0)
    
    np.testing.assert_allclose(geo.aabb_min_world, [-0.002, -0.002, 1.0])
    np.testing.assert_allclose(geo.aabb_max_world, [0.0, 0.0, 1.0])


def test_invalid_shapes(dummy_scene):
    mask, depth_m, intrinsics, pose = dummy_scene
    
    # Make mask wrong shape
    wrong_mask = np.zeros((11, 10))
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_object_geometry(wrong_mask, depth_m, intrinsics, pose)
        
    # Make mask wrong dimensions
    wrong_dim = np.zeros((10, 10, 3))
    with pytest.raises(ValueError, match="must be 2D"):
        compute_object_geometry(wrong_dim, depth_m, intrinsics, pose)
