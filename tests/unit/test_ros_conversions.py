"""Unit tests for ROS 2 conversion utilities and extended FramePacket."""

from types import SimpleNamespace
import numpy as np
import pytest

from scene_graph.data.frame_packet import FramePacket, IMUSample
from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.geometry.reference_frame import RelationReferenceFrame
from scene_graph.config import SceneGraphConfig

# Import conversion utilities
import sys
from pathlib import Path
WS_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WS_ROOT / "ros2_ws" / "src" / "scene_graph_ros"))

from scene_graph_ros.ros_conversions import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    transform_to_matrix,
    camera_info_to_intrinsics,
    ros_image_to_numpy,
)
from scene_graph_ros.config_loader import load_scene_graph_config


def test_quaternion_rotation_roundtrip():
    """Verify quaternion and rotation matrix conversion roundtrip."""
    # Identity
    R_id = quaternion_to_rotation_matrix(0.0, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(R_id, np.eye(3), atol=1e-7)

    qx, qy, qz, qw = rotation_matrix_to_quaternion(np.eye(3))
    assert pytest.approx(qw, abs=1e-6) == 1.0
    assert pytest.approx(qx, abs=1e-6) == 0.0
    assert pytest.approx(qy, abs=1e-6) == 0.0
    assert pytest.approx(qz, abs=1e-6) == 0.0

    # 90-degree yaw rotation (around Z)
    theta = np.pi / 2.0
    R_yaw = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,           0.0,            1.0],
    ])
    q_yaw = rotation_matrix_to_quaternion(R_yaw)
    R_recon = quaternion_to_rotation_matrix(*q_yaw)
    np.testing.assert_allclose(R_recon, R_yaw, atol=1e-6)
    np.testing.assert_allclose(R_recon.T @ R_recon, np.eye(3), atol=1e-6)
    assert pytest.approx(np.linalg.det(R_recon), abs=1e-6) == 1.0


def test_transform_to_matrix():
    """Verify conversion from TransformStamped / Transform mock to 4x4 matrix."""
    mock_transform = SimpleNamespace(
        translation=SimpleNamespace(x=1.2, y=-0.5, z=2.1),
        rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    mock_msg = SimpleNamespace(transform=mock_transform)

    T = transform_to_matrix(mock_msg)
    assert T.shape == (4, 4)
    assert T.dtype == np.float64
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-7)
    np.testing.assert_allclose(T[:3, 3], np.array([1.2, -0.5, 2.1]), atol=1e-7)
    assert T[3, 3] == 1.0


def test_camera_info_to_intrinsics():
    """Verify extraction of CameraIntrinsics from CameraInfo."""
    mock_cam_info = SimpleNamespace(
        width=640,
        height=480,
        k=[
            525.0, 0.0, 319.5,
            0.0, 525.0, 239.5,
            0.0, 0.0, 1.0,
        ],
        p=[],
    )

    intrinsics = camera_info_to_intrinsics(mock_cam_info)
    assert intrinsics.fx == 525.0
    assert intrinsics.fy == 525.0
    assert intrinsics.cx == 319.5
    assert intrinsics.cy == 239.5
    assert intrinsics.width == 640
    assert intrinsics.height == 480


def test_ros_image_to_numpy_rgb():
    """Verify RGB and BGR image decoding to RGB (H, W, 3) uint8."""
    # Test rgb8
    rgb_data = np.zeros((10, 20, 3), dtype=np.uint8)
    rgb_data[0, 0] = [255, 128, 64]
    mock_img_rgb = SimpleNamespace(
        height=10,
        width=20,
        encoding="rgb8",
        is_bigendian=0,
        step=20 * 3,
        data=rgb_data.tobytes(),
    )
    arr_rgb = ros_image_to_numpy(mock_img_rgb)
    assert arr_rgb.shape == (10, 20, 3)
    assert arr_rgb.dtype == np.uint8
    np.testing.assert_array_equal(arr_rgb[0, 0], [255, 128, 64])

    # Test bgr8 conversion to rgb
    bgr_data = np.zeros((10, 20, 3), dtype=np.uint8)
    bgr_data[0, 0] = [64, 128, 255]  # BGR
    mock_img_bgr = SimpleNamespace(
        height=10,
        width=20,
        encoding="bgr8",
        is_bigendian=0,
        step=20 * 3,
        data=bgr_data.tobytes(),
    )
    arr_bgr = ros_image_to_numpy(mock_img_bgr)
    assert arr_bgr.shape == (10, 20, 3)
    assert arr_bgr.dtype == np.uint8
    np.testing.assert_array_equal(arr_bgr[0, 0], [255, 128, 64])  # Converted to RGB


def test_ros_image_to_numpy_depth():
    """Verify raw depth is preserved (uint16 and float32) without scaling."""
    depth_u16 = np.arange(100, dtype=np.uint16).reshape((10, 10))
    mock_depth_16uc1 = SimpleNamespace(
        height=10,
        width=10,
        encoding="16uc1",
        is_bigendian=0,
        step=10 * 2,
        data=depth_u16.tobytes(),
    )
    arr_depth = ros_image_to_numpy(mock_depth_16uc1)
    assert arr_depth.shape == (10, 10)
    assert arr_depth.dtype == np.uint16
    np.testing.assert_array_equal(arr_depth, depth_u16)

    # Test uppercase 16UC1 as well
    mock_depth_16UC1 = SimpleNamespace(
        height=10,
        width=10,
        encoding="16UC1",
        is_bigendian=0,
        step=10 * 2,
        data=depth_u16.tobytes(),
    )
    arr_depth_UC = ros_image_to_numpy(mock_depth_16UC1)
    np.testing.assert_array_equal(arr_depth_UC, depth_u16)

    # Test 32fc1
    depth_f32 = np.linspace(0.5, 5.0, 100, dtype=np.float32).reshape((10, 10))
    mock_depth_32fc1 = SimpleNamespace(
        height=10,
        width=10,
        encoding="32fc1",
        is_bigendian=0,
        step=10 * 4,
        data=depth_f32.tobytes(),
    )
    arr_f32 = ros_image_to_numpy(mock_depth_32fc1)
    assert arr_f32.shape == (10, 10)
    assert arr_f32.dtype == np.float32
    np.testing.assert_allclose(arr_f32, depth_f32)


def test_frame_packet_imu_window():
    """Verify FramePacket supports optional windowed IMU samples."""
    intrinsics = CameraIntrinsics(fx=525, fy=525, cx=320, cy=240, width=640, height=480)
    depth_model = DepthModel(scale=5000.0)

    # Packet without IMU
    packet_no_imu = FramePacket(
        frame_index=0,
        timestamp=100.0,
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        depth=None,
        world_T_camera=None,
        camera_intrinsics=intrinsics,
        depth_model=depth_model,
    )
    assert not packet_no_imu.has_imu
    assert packet_no_imu.imu_samples == ()

    # Packet with window of IMU samples
    imu1 = IMUSample(timestamp=100.01, accel=np.array([0, 0, 9.81]), gyro=np.array([0.01, 0, 0]))
    imu2 = IMUSample(timestamp=100.02, accel=np.array([0.1, 0, 9.80]), gyro=np.array([0.02, 0, 0]))

    packet_with_imu = FramePacket(
        frame_index=1,
        timestamp=100.033,
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        depth=None,
        world_T_camera=None,
        camera_intrinsics=intrinsics,
        depth_model=depth_model,
        imu_samples=(imu1, imu2),
    )
    assert packet_with_imu.has_imu
    assert len(packet_with_imu.imu_samples) == 2
    assert packet_with_imu.imu_samples[0].timestamp == 100.01
    assert packet_with_imu.imu_samples[1].accel[2] == 9.80


def test_config_loader_deterministic():
    """Verify deterministic config loading with default and override YAMLs."""
    cfg = load_scene_graph_config("configs/default.yaml")
    assert cfg is not None
    assert cfg.reference_frame is not None
    assert cfg.reference_frame.type == "world"
    assert tuple(cfg.reference_frame.up_axis) == (0.0, 0.0, 1.0)
    assert tuple(cfg.reference_frame.heading_axis) == (1.0, 0.0, 0.0)

    # Test override
    cfg_tum = load_scene_graph_config("configs/tum_fr1_desk.yaml")
    assert cfg_tum.camera.fx == 525.0
    assert cfg_tum.depth.scale == 5000.0
