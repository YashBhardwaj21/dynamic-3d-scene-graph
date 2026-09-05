"""Unit tests for point cloud utilities (Substage 1.8)."""

import numpy as np
import pytest

from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.point_cloud import (
    compute_object_bbox_world,
    compute_object_centroid_world,
    compute_object_points_world,
)
from scene_graph.geometry.transforms import pose_to_transform


@pytest.fixture
def dummy_scene():
    """Setup a small synthetic mask, depth map, and pose for testing."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    depth = np.zeros((10, 10), dtype=np.uint16)
    
    # Create a 2x2 object in the mask
    mask[4:6, 4:6] = 255
    # Set depth values to 5000 (1.0 meter)
    depth[4:6, 4:6] = 5000
    
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=5.0, cy=5.0)
    
    # Identity pose
    pose = pose_to_transform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    
    return mask, depth, intrinsics, pose


def test_compute_object_points_world(dummy_scene):
    mask, depth, intrinsics, pose = dummy_scene
    
    points = compute_object_points_world(mask, depth, intrinsics, pose)
    
    assert points is not None
    assert points.shape == (4, 3)
    
    # Expected: z = 1.0
    # u, v range is [4, 5] x [4, 5]
    # x = (u - cx) * z / fx = (4 - 5) / 500 = -0.002
    np.testing.assert_array_equal(points[:, 2], 1.0)


def test_compute_object_centroid_world(dummy_scene):
    mask, depth, intrinsics, pose = dummy_scene
    
    centroid = compute_object_centroid_world(mask, depth, intrinsics, pose)
    
    assert centroid is not None
    assert centroid.shape == (3,)
    assert centroid[2] == pytest.approx(1.0)


def test_compute_object_bbox_world(dummy_scene):
    mask, depth, intrinsics, pose = dummy_scene
    
    bbox = compute_object_bbox_world(mask, depth, intrinsics, pose)
    
    assert bbox is not None
    bbox_min, bbox_max = bbox
    assert bbox_min.shape == (3,)
    assert bbox_max.shape == (3,)
    assert bbox_min[2] == pytest.approx(1.0)
    assert bbox_max[2] == pytest.approx(1.0)
