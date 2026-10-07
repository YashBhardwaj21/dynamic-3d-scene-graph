"""Unit tests for Camera Geometry, Optical/Body Coordinate Conventions, and 3D Projection (Stage 5)."""

import numpy as np
import pytest

from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.geometry.transforms import (
    pose_to_transform,
    transform_points,
    invert_se3_transform,
    transform_points_inverse,
    optical_to_body_rotation,
    body_to_optical_rotation,
    optical_to_body_transform,
    body_to_optical_transform,
)
from scene_graph.geometry.point_cloud import (
    compute_object_geometry,
    GeometryStatus,
)


def test_intrinsics_validation():
    """Verify CameraIntrinsics strictly validates physical parameters and image boundaries."""
    # Valid intrinsics
    cam = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
    assert cam.fx == 600.0
    assert cam.width == 640

    # Non-positive focal lengths
    with pytest.raises(ValueError, match="Focal lengths .* must be > 0"):
        CameraIntrinsics(fx=0.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
    with pytest.raises(ValueError, match="Focal lengths .* must be > 0"):
        CameraIntrinsics(fx=-10.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
    with pytest.raises(ValueError, match="Focal lengths .* must be > 0"):
        CameraIntrinsics(fx=600.0, fy=-5.0, cx=320.0, cy=240.0, width=640, height=480)

    # Invalid image dimensions
    with pytest.raises(ValueError, match="Image dimensions .* must be > 0"):
        CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=0, height=480)
    with pytest.raises(ValueError, match="Image dimensions .* must be > 0"):
        CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=-1)

    # Principal point out of frame boundaries
    with pytest.raises(ValueError, match="lies outside image boundaries"):
        CameraIntrinsics(fx=600.0, fy=600.0, cx=-1.0, cy=240.0, width=640, height=480)
    with pytest.raises(ValueError, match="lies outside image boundaries"):
        CameraIntrinsics(fx=600.0, fy=600.0, cx=641.0, cy=240.0, width=640, height=480)
    with pytest.raises(ValueError, match="lies outside image boundaries"):
        CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=-0.5, width=640, height=480)
    with pytest.raises(ValueError, match="lies outside image boundaries"):
        CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=481.0, width=640, height=480)


def test_camera_projection_roundtrip_scalar():
    """Verify 3D <-> 2D scalar projection roundtrip is exact within 1e-10 meters."""
    cam = CameraIntrinsics(fx=525.0, fy=525.0, cx=319.5, cy=239.5, width=640, height=480)
    
    # Test multiple pixel coordinates with varying depths
    test_cases = [
        (319.5, 239.5, 1.0),
        (100.0, 50.0, 2.5),
        (550.0, 420.0, 0.75),
        (10.0, 10.0, 8.0),
    ]
    
    for u_orig, v_orig, z_orig in test_cases:
        # 2D -> 3D
        x, y, z = cam.pixel_to_camera(u_orig, v_orig, z_orig)
        assert z == pytest.approx(z_orig, abs=1e-12)
        
        # 3D -> 2D
        u_proj, v_proj = cam.camera_to_pixel(x, y, z)
        assert u_proj == pytest.approx(u_orig, abs=1e-10)
        assert v_proj == pytest.approx(v_orig, abs=1e-10)


def test_camera_projection_roundtrip_vectorized():
    """Verify vectorized 3D <-> 2D projection roundtrip on arbitrary synthetic points."""
    cam = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)
    
    rng = np.random.default_rng(42)
    N = 1000
    u_orig = rng.uniform(0.0, 640.0, size=N)
    v_orig = rng.uniform(0.0, 480.0, size=N)
    z_orig = rng.uniform(0.2, 8.0, size=N)
    
    # Vectorized 2D -> 3D
    pts_cam = cam.pixels_to_camera(u_orig, v_orig, z_orig)
    assert pts_cam.shape == (N, 3)
    np.testing.assert_allclose(pts_cam[:, 2], z_orig, atol=1e-12)
    
    # Vectorized 3D -> 2D
    u_proj, v_proj = cam.cameras_to_pixels(pts_cam)
    np.testing.assert_allclose(u_proj, u_orig, atol=1e-10)
    np.testing.assert_allclose(v_proj, v_orig, atol=1e-10)


def test_camera_projection_invalid_depth():
    """Verify points behind or at the focal plane (z <= 0) are strictly rejected."""
    cam = CameraIntrinsics(fx=525.0, fy=525.0, cx=319.5, cy=239.5, width=640, height=480)
    
    with pytest.raises(ValueError, match="Invalid camera depth"):
        cam.camera_to_pixel(1.0, 1.0, 0.0)
        
    with pytest.raises(ValueError, match="Invalid camera depth"):
        cam.camera_to_pixel(1.0, 1.0, -1.5)
        
    with pytest.raises(ValueError, match="finite positive depth"):
        cam.cameras_to_pixels(np.array([[1.0, 2.0, -0.5], [1.0, 2.0, 1.0]]))


def test_depth_clipping_bounds():
    """Verify depth filtering rejects values outside [min_depth_m, max_depth_m] and non-finite values."""
    cam = CameraIntrinsics(fx=500.0, fy=500.0, cx=5.0, cy=5.0, width=10, height=10)
    pose = np.eye(4)
    
    mask = np.ones((10, 10), dtype=np.uint8)
    depth = np.full((10, 10), 1.0, dtype=np.float64)
    
    # Near clipping violation (< 0.10m)
    depth[0:5, 0:5] = 0.05
    # Far clipping violation (> 10.0m)
    depth[0:5, 5:10] = 12.0
    # Non-finite values
    depth[5:7, 0:2] = np.nan
    depth[5:7, 2:4] = np.inf
    depth[5:7, 4:6] = -1.0
    
    # Remainder: points with valid depth = 1.0
    geom = compute_object_geometry(
        mask=mask,
        depth_m=depth,
        intrinsics=cam,
        pose=pose,
        min_valid_points=10,
        min_depth_m=0.10,
        max_depth_m=10.0,
    )
    assert geom is not None
    assert geom.status == GeometryStatus.VALID
    # All surviving points must have z in [0.10, 10.0]
    assert np.all(geom.points_camera[:, 2] >= 0.10)
    assert np.all(geom.points_camera[:, 2] <= 10.0)
    # The corrupted regions (50 near/far + 12 invalid = 62 pixels) were stripped
    assert geom.valid_point_count == 38


def test_mad_and_spatial_outlier_filtering():
    """Verify MAD and 3D spatial outlier rejection strips background wall and flying points."""
    cam = CameraIntrinsics(fx=500.0, fy=500.0, cx=20.0, cy=20.0, width=40, height=40)
    pose = np.eye(4)
    
    mask = np.zeros((40, 40), dtype=np.uint8)
    depth = np.zeros((40, 40), dtype=np.float64)
    
    # Foreground object: 10x10 patch at z = 1.0 m (100 pixels)
    mask[10:20, 10:20] = 1
    depth[10:20, 10:20] = 1.0
    
    # Background leakage: 5x5 patch at z = 3.5 m (25 pixels background wall)
    mask[20:25, 20:25] = 1
    depth[20:25, 20:25] = 3.5
    
    geom = compute_object_geometry(
        mask=mask,
        depth_m=depth,
        intrinsics=cam,
        pose=pose,
        min_valid_points=30,
        mad_k=2.5,
    )
    assert geom is not None
    assert geom.status == GeometryStatus.VALID
    # The foreground object at 1.0m should be retained, background at 3.5m stripped
    assert geom.valid_point_count == 100
    np.testing.assert_allclose(geom.centroid_world[2], 1.0, atol=1e-3)


def test_optical_to_body_transform_rep103():
    """Verify standard ROS REP-103 coordinate conversion between Optical and Body frames."""
    R_opt_to_body = optical_to_body_rotation()
    R_body_to_opt = body_to_optical_rotation()
    
    # Must be valid orthonormal rotation matrices
    np.testing.assert_allclose(R_opt_to_body.T @ R_opt_to_body, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(R_body_to_opt.T @ R_body_to_opt, np.eye(3), atol=1e-12)
    assert np.linalg.det(R_opt_to_body) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.det(R_body_to_opt) == pytest.approx(1.0, abs=1e-12)
    
    # Inverses
    np.testing.assert_allclose(R_opt_to_body @ R_body_to_opt, np.eye(3), atol=1e-12)
    
    # REP-103 physical axis mappings:
    # Optical: +Z is forward -> Body: +X is forward
    z_opt = np.array([0.0, 0.0, 1.0])
    x_body = R_opt_to_body @ z_opt
    np.testing.assert_allclose(x_body, [1.0, 0.0, 0.0], atol=1e-12)
    
    # Optical: +X is right -> Body: +Y is left, so -Y is right
    x_opt = np.array([1.0, 0.0, 0.0])
    y_body = R_opt_to_body @ x_opt
    np.testing.assert_allclose(y_body, [0.0, -1.0, 0.0], atol=1e-12)
    
    # Optical: +Y is down -> Body: +Z is up, so -Z is down
    y_opt = np.array([0.0, 1.0, 0.0])
    z_body = R_opt_to_body @ y_opt
    np.testing.assert_allclose(z_body, [0.0, 0.0, -1.0], atol=1e-12)


def test_se3_inversion_and_roundtrip():
    """Verify invert_se3_transform and transform_points_inverse exact mathematical properties."""
    # Arbitrary rigid transform
    T = pose_to_transform(1.5, -2.3, 0.8, 0.1, 0.2, 0.3, 0.9)
    T_inv = invert_se3_transform(T)
    
    # T @ T_inv == I
    np.testing.assert_allclose(T @ T_inv, np.eye(4), atol=1e-10)
    np.testing.assert_allclose(T_inv @ T, np.eye(4), atol=1e-10)
    
    # Roundtrip point transformation
    pts_cam = np.array([
        [0.5, -0.2, 1.8],
        [1.0, 0.4, 2.5],
        [-0.3, -0.7, 3.2],
    ], dtype=np.float64)
    
    pts_world = transform_points(T, pts_cam)
    pts_cam_recovered = transform_points_inverse(T, pts_world)
    np.testing.assert_allclose(pts_cam_recovered, pts_cam, atol=1e-12)


def test_obb_right_handed_basis():
    """Verify OBB computation always yields a right-handed basis: det(obb_axes) == +1.0."""
    cam = CameraIntrinsics(fx=500.0, fy=500.0, cx=50.0, cy=50.0, width=100, height=100)
    pose = np.eye(4)
    
    # Create anisotropic 3D point cluster with a known orientation
    mask = np.zeros((100, 100), dtype=np.uint8)
    depth = np.zeros((100, 100), dtype=np.float64)
    
    # 20x10 rectangle
    mask[40:60, 45:55] = 1
    depth[40:60, 45:55] = 1.2
    
    geom = compute_object_geometry(
        mask=mask,
        depth_m=depth,
        intrinsics=cam,
        pose=pose,
        min_valid_points=20,
    )
    assert geom is not None
    assert geom.status == GeometryStatus.VALID
    assert geom.obb_axes_world is not None
    
    # Verify right-handed orthonormal basis
    det_axes = np.linalg.det(geom.obb_axes_world)
    assert det_axes == pytest.approx(1.0, abs=1e-7)
    np.testing.assert_allclose(geom.obb_axes_world.T @ geom.obb_axes_world, np.eye(3), atol=1e-7)
