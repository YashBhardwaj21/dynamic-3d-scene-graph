"""Unit tests for Sensor-Realistic Measurement Noise and Covariance Propagation (Stage 6)."""

import numpy as np
import pytest

from scene_graph.config import SceneGraphConfig, SensorConfig
from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.noise_model import (
    IsotropicNoiseModel,
    StereoDepthNoiseModel,
    QuadraticDepthNoiseModel,
    create_noise_model_from_config,
)
from scene_graph.geometry.point_cloud import compute_object_geometry, GeometryStatus
from scene_graph.geometry.transforms import pose_to_transform
from scene_graph.tracking.causal_tracker import CausalTracker
from scene_graph.tracking.state import KalmanState
from scene_graph.perception.observation import Observation


@pytest.fixture
def standard_intrinsics():
    return CameraIntrinsics(fx=525.0, fy=525.0, cx=320.0, cy=240.0, width=640, height=480)


def test_isotropic_noise_model_invariance(standard_intrinsics):
    """Verify IsotropicNoiseModel produces spherical covariance invariant under rotation."""
    model = IsotropicNoiseModel(std_m=0.03)
    pts = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)

    # In camera frame
    cov_cam = model.estimate_covariance(pts, standard_intrinsics)
    np.testing.assert_allclose(cov_cam, np.eye(3) * (0.03 ** 2))

    # In rotated world frame (arbitrary 3D rotation)
    R = pose_to_transform(0, 0, 0, 0.2, 0.4, 0.1, 0.894)[:3, :3]
    cov_world = model.estimate_covariance(pts, standard_intrinsics, R_world_camera=R)
    # R * (s^2 * I) * R^T = s^2 * I
    np.testing.assert_allclose(cov_world, np.eye(3) * (0.03 ** 2), atol=1e-10)


def test_stereo_noise_model_anisotropy_and_rotation(standard_intrinsics):
    """Verify StereoDepthNoiseModel aligns highest uncertainty along camera viewing ray."""
    baseline = 0.095
    model = StereoDepthNoiseModel(
        baseline_m=baseline,
        subpixel_disparity_std=0.1,
        pixel_noise_std=0.5,
    )
    # Object at 2 meters along camera +Z axis
    pts = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)

    # Camera frame: Z-axis uncertainty >> lateral X and Y uncertainty
    cov_cam = model.estimate_covariance(pts, standard_intrinsics)
    var_x = cov_cam[0, 0]
    var_y = cov_cam[1, 1]
    var_z = cov_cam[2, 2]

    assert var_z > var_x * 10.0
    assert var_z > var_y * 10.0
    assert cov_cam[0, 1] == 0.0
    assert cov_cam[0, 2] == 0.0

    # 90-degree rotation around Y axis:
    # Camera +Z (optical forward) maps to World +X
    R_y90 = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float64)

    cov_world = model.estimate_covariance(pts, standard_intrinsics, R_world_camera=R_y90)

    # In world frame, the large axial variance must now be along world X
    np.testing.assert_allclose(cov_world[0, 0], var_z, atol=1e-10)
    np.testing.assert_allclose(cov_world[1, 1], var_y, atol=1e-10)
    np.testing.assert_allclose(cov_world[2, 2], var_x, atol=1e-10)

    # Must be strictly symmetric positive-definite
    np.testing.assert_allclose(cov_world, cov_world.T, atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(cov_world)
    assert np.all(eigenvalues > 0.0)


def test_stereo_noise_model_axial_noise_floor(standard_intrinsics):
    """Verify axial_noise_floor_m prevents unphysically zero variance at close range."""
    floor_m = 0.005
    model = StereoDepthNoiseModel(
        baseline_m=0.095,
        subpixel_disparity_std=0.1,
        pixel_noise_std=0.5,
        axial_noise_floor_m=floor_m,
    )

    pts_close = np.array([[0.0, 0.0, 0.1]], dtype=np.float64)
    cov = model.estimate_covariance(pts_close, standard_intrinsics)
    std_z = np.sqrt(cov[2, 2])

    assert std_z >= floor_m


def test_quadratic_depth_noise_model(standard_intrinsics):
    """Verify QuadraticDepthNoiseModel follows sigma_z(z) = alpha * z^2 + sigma_0."""
    alpha = 0.002
    sigma_0 = 0.004
    model = QuadraticDepthNoiseModel(alpha=alpha, sigma_0=sigma_0, pixel_noise_std=0.5)

    pts_1m = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    pts_3m = np.array([[0.0, 0.0, 3.0]], dtype=np.float64)

    cov_1m = model.estimate_covariance(pts_1m, standard_intrinsics)
    cov_3m = model.estimate_covariance(pts_3m, standard_intrinsics)

    expected_sigma_z_1m = alpha * (1.0 ** 2) + sigma_0
    expected_sigma_z_3m = alpha * (3.0 ** 2) + sigma_0

    np.testing.assert_allclose(np.sqrt(cov_1m[2, 2]), expected_sigma_z_1m, rtol=1e-5)
    np.testing.assert_allclose(np.sqrt(cov_3m[2, 2]), expected_sigma_z_3m, rtol=1e-5)

    # Verify symmetry and positive definiteness under rotation
    R = pose_to_transform(0, 0, 0, 0.1, 0.3, 0.5, 0.8)[:3, :3]
    cov_rot = model.estimate_covariance(pts_3m, standard_intrinsics, R_world_camera=R)
    np.testing.assert_allclose(cov_rot, cov_rot.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(cov_rot) > 0.0)


def test_point_cloud_covariance_symmetry_and_definiteness(standard_intrinsics):
    """Verify compute_object_geometry produces strictly symmetric positive-definite covariance."""
    model = StereoDepthNoiseModel(baseline_m=0.095, subpixel_disparity_std=0.1)

    mask = np.ones((20, 20), dtype=np.uint8)
    depth = np.full((20, 20), 2.5, dtype=np.float64)
    pose = pose_to_transform(1.0, 2.0, 0.5, 0.2, 0.1, 0.3, 0.9)

    geom = compute_object_geometry(
        mask=mask,
        depth_m=depth,
        intrinsics=standard_intrinsics,
        pose=pose,
        noise_model=model,
    )
    assert geom.status == GeometryStatus.VALID
    cov = geom.position_covariance_world

    assert cov.shape == (3, 3)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > 0.0)


def test_kalman_state_matrix_covariance_initialization():
    """Verify KalmanState correctly accepts and preserves a full 3x3 positive-definite covariance."""
    pos = np.array([1.0, 2.0, 3.0])
    
    # Anisotropic covariance (e.g. from depth sensor)
    P_pos_init = np.array([
        [0.001, 0.0002, 0.0],
        [0.0002, 0.002, 0.0],
        [0.0,    0.0,    0.05],
    ], dtype=np.float64)

    state = KalmanState(pos, initial_cov_pos=P_pos_init, initial_cov_vel=1.0)

    np.testing.assert_allclose(state.position, pos)
    np.testing.assert_allclose(state.position_covariance, P_pos_init, atol=1e-12)
    np.testing.assert_allclose(state.velocity, [0.0, 0.0, 0.0])


def test_tracker_covariance_propagation_with_distance(standard_intrinsics):
    """Verify CausalTracker reflects higher uncertainty for distant observations in Kalman state."""
    config = SceneGraphConfig.from_files("configs/default.yaml")
    tracker = CausalTracker(config)

    # Simulate two observations:
    # Obs A: close at z=1.0m (low uncertainty)
    # Obs B: distant at z=5.0m (high uncertainty)
    pose = np.eye(4)
    stereo_model = StereoDepthNoiseModel(baseline_m=0.095, subpixel_disparity_std=0.1)

    # Geometry for Obs A (close)
    mask_a = np.ones((20, 20), dtype=np.uint8)
    depth_a = np.full((20, 20), 1.0, dtype=np.float64)
    geom_a = compute_object_geometry(
        mask=mask_a, depth_m=depth_a, intrinsics=standard_intrinsics, pose=pose, noise_model=stereo_model
    )

    # Geometry for Obs B (distant)
    mask_b = np.ones((20, 20), dtype=np.uint8)
    depth_b = np.full((20, 20), 5.0, dtype=np.float64)
    geom_b = compute_object_geometry(
        mask=mask_b, depth_m=depth_b, intrinsics=standard_intrinsics, pose=pose, noise_model=stereo_model
    )

    obs_close = Observation(
        obs_id="obs_close",
        frame_index=0,
        timestamp=0.0,
        class_name="cup",
        confidence=0.9,
        bbox_xyxy=np.array([100, 100, 120, 120]),
        mask_rle=None,
        object_geometry=geom_a,
    )
    obs_distant = Observation(
        obs_id="obs_distant",
        frame_index=0,
        timestamp=0.0,
        class_name="bottle",
        confidence=0.9,
        bbox_xyxy=np.array([400, 400, 420, 420]),
        mask_rle=None,
        object_geometry=geom_b,
    )

    # Track both
    tracker.update([obs_close], frame_index=0, timestamp=0.0)
    tracker.update([obs_distant], frame_index=0, timestamp=0.0)

    track_close = [t for t in tracker.tracks.values() if t.class_name == "cup"][0]
    track_distant = [t for t in tracker.tracks.values() if t.class_name == "bottle"][0]

    # Distant track's position covariance along Z must be strictly larger than close track's
    cov_close_z = track_close.position_covariance_world[2, 2]
    cov_distant_z = track_distant.position_covariance_world[2, 2]

    assert cov_distant_z > cov_close_z * 10.0


def test_create_noise_model_from_config_quadratic():
    """Verify create_noise_model_from_config instantiates QuadraticDepthNoiseModel."""
    config = SceneGraphConfig.from_files("configs/default.yaml")
    config.sensor = SensorConfig(
        name="test_rgbd",
        measurement_model="quadratic",
        quadratic_alpha=0.002,
        quadratic_sigma_0=0.005,
    )

    model = create_noise_model_from_config(config)
    assert isinstance(model, QuadraticDepthNoiseModel)
    assert model.alpha == 0.002
    assert model.sigma_0 == 0.005
