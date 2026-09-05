"""Unit tests for geometry module (Substages 1.4, 1.5)."""

import numpy as np
import pytest

from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.transforms import (
    pose_to_transform,
    quaternion_to_matrix,
    transform_points,
)


def test_camera_intrinsics_projection():
    """Regression test for central pixel projection.
    u=319.5, v=239.5, z=1.0 -> x=0, y=0, z=1
    """
    cam = CameraIntrinsics(fx=525.0, fy=525.0, cx=319.5, cy=239.5)
    x, y, z = cam.pixel_to_camera(319.5, 239.5, 1.0)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(1.0)


def test_quaternion_to_matrix_identity():
    """Verify identity quaternion converts to identity matrix."""
    R = quaternion_to_matrix(0, 0, 0, 1)
    np.testing.assert_allclose(R, np.eye(3))


def test_quaternion_to_matrix_rotation():
    """Verify 90 degree rotation about X axis."""
    qx, qy, qz, qw = np.sin(np.pi / 4), 0, 0, np.cos(np.pi / 4)
    R = quaternion_to_matrix(qx, qy, qz, qw)
    expected = [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0]
    ]
    np.testing.assert_allclose(R, expected, atol=1e-7)


def test_pose_to_transform():
    """Verify pose is converted to 4x4 SE(3) matrix."""
    T = pose_to_transform(1.0, 2.0, 3.0, 0, 0, 0, 1)
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(T[:3, :3], np.eye(3))
    np.testing.assert_allclose(T[3, :], [0.0, 0.0, 0.0, 1.0])


def test_transform_points():
    """Verify Nx3 point cloud transformation."""
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    
    points_cam = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    
    points_world = transform_points(T, points_cam)
    
    expected = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 2.0, 3.0]
    ])
    assert points_world.shape == (2, 3)
    np.testing.assert_allclose(points_world, expected)


def test_transform_points_rotation():
    """Verify point cloud transformation under a nontrivial rotation (90 deg around X)."""
    # 90 degrees around X axis
    # R = [1  0  0
    #      0  0 -1
    #      0  1  0]
    qx, qy, qz, qw = np.sin(np.pi / 4), 0, 0, np.cos(np.pi / 4)
    T = pose_to_transform(1.0, 2.0, 3.0, qx, qy, qz, qw)
    
    # Point [0, 1, 0] rotated 90 around X becomes [0, 0, 1]
    # Then translated by [1, 2, 3] becomes [1, 2, 4]
    points_cam = np.array([[0.0, 1.0, 0.0]])
    points_world = transform_points(T, points_cam)
    
    expected = np.array([[1.0, 2.0, 4.0]])
    np.testing.assert_allclose(points_world, expected, atol=1e-7)


def test_transform_points_empty():
    """Verify graceful handling of empty point clouds."""
    T = np.eye(4)
    points_cam = np.empty((0, 3))
    points_world = transform_points(T, points_cam)
    assert points_world.shape == (0, 3)


from scene_graph.geometry.reference_frame import compute_alignment_from_normal

def test_alignment_transform_opposite_normal():
    """Verify correct alignment when normal is exactly opposite to target [0, 0, 1]."""
    # Use normal pointing exactly down
    normal = np.array([0.0, 0.0, -1.0])
    
    T = compute_alignment_from_normal(normal)
    R = T[:3, :3]
    
    # Verify it maps [0, 0, -1] to [0, 0, 1]
    aligned_normal = R @ normal
    
    np.testing.assert_allclose(aligned_normal, [0.0, 0.0, 1.0], atol=1e-7)
    
    # Verify R is a valid rotation matrix (R @ R.T = I and det(R) = +1)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-7)
    assert pytest.approx(np.linalg.det(R)) == 1.0
