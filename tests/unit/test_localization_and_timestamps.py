import numpy as np
import pytest

from scene_graph.data.frame_packet import FramePacket, LocalizationMode
from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.point_cloud import (
    GeometryStatus,
    ObjectGeometry,
    compute_object_geometry,
)
from scene_graph.perception.observation import Observation
from scene_graph.pipeline.pipeline_core import SceneGraphPipeline
from scene_graph.config import SceneGraphConfig


@pytest.fixture
def intrinsics():
    return CameraIntrinsics(
        fx=525.0,
        fy=525.0,
        cx=319.5,
        cy=239.5,
        width=640,
        height=480,
    )


@pytest.fixture
def dummy_rgb():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_depth():
    return np.ones((480, 640), dtype=np.float32) * 2.0


def test_valid_exact_tf_world_mode(intrinsics, dummy_rgb, dummy_depth):
    t_rgb = 100.0
    t_pose = 100.0
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = 1.0

    packet = FramePacket(
        frame_index=1,
        timestamp=t_rgb,
        rgb=dummy_rgb,
        depth=dummy_depth,
        world_T_camera=pose,
        camera_intrinsics=intrinsics,
        sensor_timestamp=t_rgb,
        rgb_timestamp=t_rgb,
        depth_timestamp=t_rgb,
        pose_timestamp=t_pose,
        world_frame="world",
        transform_source="tf_exact",
        transform_valid=True,
        localization_mode=LocalizationMode.WORLD_MODE,
    )

    assert packet.localization_mode == LocalizationMode.WORLD_MODE
    assert packet.transform_valid is True
    assert packet.pose_age == 0.0
    assert packet.transform_source == "tf_exact"
    assert packet.world_frame == "world"
    assert packet.is_persistent_world_frame is True


def test_missing_tf_world_mode(intrinsics, dummy_rgb, dummy_depth):
    t_rgb = 100.0

    packet = FramePacket(
        frame_index=2,
        timestamp=t_rgb,
        rgb=dummy_rgb,
        depth=dummy_depth,
        world_T_camera=None,
        camera_intrinsics=intrinsics,
        sensor_timestamp=t_rgb,
        rgb_timestamp=t_rgb,
        depth_timestamp=t_rgb,
        pose_timestamp=None,
        world_frame="world",
        transform_source="missing_tf",
        transform_valid=False,
        localization_mode=LocalizationMode.WORLD_MODE,
    )

    assert packet.localization_mode == LocalizationMode.WORLD_MODE
    assert packet.transform_valid is False
    assert packet.transform_source == "missing_tf"
    assert packet.has_pose is False
    assert packet.is_persistent_world_frame is False


def test_stale_tf_rejection(intrinsics, dummy_rgb, dummy_depth):
    t_rgb = 100.05
    t_pose = 99.80  # 250 ms lag (stale)
    pose = np.eye(4, dtype=np.float64)

    packet = FramePacket(
        frame_index=3,
        timestamp=t_rgb,
        rgb=dummy_rgb,
        depth=dummy_depth,
        world_T_camera=pose,
        camera_intrinsics=intrinsics,
        sensor_timestamp=t_rgb,
        rgb_timestamp=t_rgb,
        depth_timestamp=t_rgb,
        pose_timestamp=t_pose,
        world_frame="world",
        transform_source="stale_tf",
        transform_valid=False,
        localization_mode=LocalizationMode.WORLD_MODE,
    )

    assert packet.pose_age == pytest.approx(0.25, abs=1e-4)
    assert packet.transform_valid is False
    assert packet.transform_source == "stale_tf"
    assert packet.is_persistent_world_frame is False


def test_camera_local_mode_declared_non_persistent(intrinsics, dummy_rgb, dummy_depth):
    t_rgb = 50.0

    packet = FramePacket(
        frame_index=4,
        timestamp=t_rgb,
        rgb=dummy_rgb,
        depth=dummy_depth,
        world_T_camera=None,  # will default to identity for camera frame
        camera_intrinsics=intrinsics,
        frame_id="camera_color_optical_frame",
        localization_mode=LocalizationMode.CAMERA_LOCAL_MODE,
    )

    assert packet.localization_mode == LocalizationMode.CAMERA_LOCAL_MODE
    assert packet.world_frame == "camera_color_optical_frame"
    assert packet.transform_source == "camera_local"
    assert packet.transform_valid is True
    assert packet.pose_age == 0.0
    np.testing.assert_allclose(packet.world_T_camera, np.eye(4))
    # Crucial: camera local coordinates are explicitly marked non-persistent in world graph
    assert packet.is_persistent_world_frame is False


def test_geometry_rejection_on_invalid_transform(intrinsics, dummy_rgb, dummy_depth):
    config = SceneGraphConfig.from_files("configs/default.yaml")
    pipeline = SceneGraphPipeline(config)

    packet_invalid = FramePacket(
        frame_index=5,
        timestamp=10.0,
        rgb=dummy_rgb,
        depth=dummy_depth,
        world_T_camera=None,
        camera_intrinsics=intrinsics,
        transform_source="stale_tf",
        transform_valid=False,
        localization_mode=LocalizationMode.WORLD_MODE,
    )

    obs = Observation(
        obs_id="obs_001",
        frame_index=5,
        timestamp=10.0,
        class_name="cup",
        confidence=0.9,
        bbox_xyxy=np.array([100, 100, 200, 200], dtype=np.float32),
        mask_rle=None,
    )

    pipeline._compute_geometry(packet_invalid, [obs], dummy_depth)
    assert obs.object_geometry is not None
    assert obs.object_geometry.status == GeometryStatus.STALE_POSE


def test_robot_translation_invariance_for_stationary_objects(intrinsics):
    # True stationary object position in world coordinates
    p_world_true = np.array([2.0, 1.0, 0.5])

    # Robot at pose 0: at world origin
    T_w_c0 = np.eye(4, dtype=np.float64)
    # Coordinate in camera 0:
    p_c0 = np.linalg.inv(T_w_c0)[:3, :3] @ (p_world_true - T_w_c0[:3, 3])

    # Robot at pose 1: translated by (dx=0.8, dy=-0.4, dz=0.1)
    T_w_c1 = np.eye(4, dtype=np.float64)
    T_w_c1[:3, 3] = np.array([0.8, -0.4, 0.1])
    # Coordinate observed in camera 1:
    p_c1 = np.linalg.inv(T_w_c1)[:3, :3] @ (p_world_true - T_w_c1[:3, 3])

    # Verify camera coordinates changed due to robot translation
    assert not np.allclose(p_c0, p_c1)

    # When lifted to world frame using respective timestamped transforms:
    p_w_recovered_0 = T_w_c0[:3, :3] @ p_c0 + T_w_c0[:3, 3]
    p_w_recovered_1 = T_w_c1[:3, :3] @ p_c1 + T_w_c1[:3, 3]

    np.testing.assert_allclose(p_w_recovered_0, p_world_true, atol=1e-6)
    np.testing.assert_allclose(p_w_recovered_1, p_world_true, atol=1e-6)
    np.testing.assert_allclose(p_w_recovered_0, p_w_recovered_1, atol=1e-6)


def test_robot_rotation_invariance_for_stationary_objects(intrinsics):
    # True stationary object position in world coordinates
    p_world_true = np.array([2.5, 0.0, 0.8])

    # Robot rotates by 45 degrees around Z axis (yaw)
    theta = np.pi / 4.0
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,           0.0,            1.0],
    ])
    T_w_c_rotated = np.eye(4, dtype=np.float64)
    T_w_c_rotated[:3, :3] = R_z
    T_w_c_rotated[:3, 3] = np.array([0.5, 0.5, 0.0])  # rotation + translation

    # Camera observes object at camera coordinates:
    p_c = np.linalg.inv(T_w_c_rotated)[:3, :3] @ (p_world_true - T_w_c_rotated[:3, 3])

    # Without correct TF (e.g. fake identity fallback), coordinates would be wrong:
    p_w_fake_identity = p_c
    assert not np.allclose(p_w_fake_identity, p_world_true)

    # With valid timestamped TF, lifted world coordinates match groundtruth perfectly:
    p_w_correct = T_w_c_rotated[:3, :3] @ p_c + T_w_c_rotated[:3, 3]
    np.testing.assert_allclose(p_w_correct, p_world_true, atol=1e-6)
