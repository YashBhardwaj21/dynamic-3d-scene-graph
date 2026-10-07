import numpy as np
import pytest

from scene_graph.config import SceneGraphConfig
from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.noise_model import (
    CustomNoiseModel,
    DepthNoiseModel,
    IsotropicNoiseModel,
    StereoDepthNoiseModel,
    create_noise_model_from_config,
)
from scene_graph.geometry.point_cloud import compute_object_geometry, GeometryStatus


def test_stereo_noise_model_axial_uncertainty_scaling():
    """Asserts axial range uncertainty scales quadratically with depth under StereoDepthNoiseModel."""
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    noise_model = StereoDepthNoiseModel(
        baseline_m=0.095,
        subpixel_disparity_std=0.1,
        pixel_noise_std=0.5,
    )

    pts_1m = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    pts_4m = np.array([[0.0, 0.0, 4.0]], dtype=np.float64)

    cov_1m = noise_model.estimate_covariance(pts_1m, intrinsics)
    cov_4m = noise_model.estimate_covariance(pts_4m, intrinsics)

    # sigma_z = (z^2 / (f * B)) * sigma_d
    # sigma_z(4.0) / sigma_z(1.0) = (4.0 / 1.0)^2 = 16
    # Variance (cov[2, 2]) = sigma_z^2 -> (16)^2 = 256
    var_z_1m = cov_1m[2, 2]
    var_z_4m = cov_4m[2, 2]

    std_z_1m = np.sqrt(var_z_1m)
    std_z_4m = np.sqrt(var_z_4m)

    ratio_std = std_z_4m / std_z_1m
    assert np.isclose(ratio_std, 16.0, rtol=1e-3)

    ratio_var = var_z_4m / var_z_1m
    assert np.isclose(ratio_var, 256.0, rtol=1e-3)


def test_geometry_computation_with_different_noise_models():
    """Asserts compute_object_geometry operates identically across different noise models without branching."""
    H, W = 100, 100
    mask = np.zeros((H, W), dtype=bool)
    mask[40:60, 40:60] = True
    depth_m = np.ones((H, W), dtype=np.float32) * 2.0
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=50.0, cy=50.0, width=W, height=H)
    pose = np.eye(4)

    iso_model = IsotropicNoiseModel(std_m=0.02)
    stereo_model = StereoDepthNoiseModel(baseline_m=0.095, subpixel_disparity_std=0.1)
    custom_model = CustomNoiseModel(lambda pts, intr, R: np.eye(3) * 0.005)

    geom_iso = compute_object_geometry(
        mask=mask, depth_m=depth_m, intrinsics=intrinsics, pose=pose, noise_model=iso_model
    )
    geom_stereo = compute_object_geometry(
        mask=mask, depth_m=depth_m, intrinsics=intrinsics, pose=pose, noise_model=stereo_model
    )
    geom_custom = compute_object_geometry(
        mask=mask, depth_m=depth_m, intrinsics=intrinsics, pose=pose, noise_model=custom_model
    )

    assert geom_iso.status == GeometryStatus.VALID
    assert geom_stereo.status == GeometryStatus.VALID
    assert geom_custom.status == GeometryStatus.VALID

    assert geom_iso.position_covariance_world.shape == (3, 3)
    assert geom_stereo.position_covariance_world.shape == (3, 3)
    assert geom_custom.position_covariance_world.shape == (3, 3)

    # All should have positive eigenvalues (positive definite)
    assert np.all(np.linalg.eigvals(geom_iso.position_covariance_world) > 0)
    assert np.all(np.linalg.eigvals(geom_stereo.position_covariance_world) > 0)
    assert np.all(np.linalg.eigvals(geom_custom.position_covariance_world) > 0)


def test_realsense_d455_sensor_config_loading():
    """Asserts configs/sensors/realsense_d455.yaml loads and instantiates StereoDepthNoiseModel."""
    config = SceneGraphConfig.from_files("configs/sensors/realsense_d455.yaml")
    assert config.sensor is not None
    assert config.sensor.name == "realsense_d455"
    assert config.sensor.measurement_model == "stereo"
    assert config.sensor.baseline_m == 0.095

    noise_model = create_noise_model_from_config(config)
    assert isinstance(noise_model, StereoDepthNoiseModel)
    assert noise_model.baseline_m == 0.095
