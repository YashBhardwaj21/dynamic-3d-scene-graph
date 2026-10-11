"""Unit tests for Stage 1 Depth Scale and Conversion Invariants."""

import numpy as np
import pytest

from scene_graph.data.sensor_frame import SensorFrame, StreamStatus
from scene_graph.data.timestamp import Timestamp, TimestampDomain
from scene_graph.geometry.camera import CameraIntrinsics


def test_d455_runtime_depth_scale_conversion():
    """Verify D455 conversion: raw uint16 * runtime depth_scale = meters."""
    raw_depth = np.array([[1000, 2500], [5000, 0]], dtype=np.uint16)
    runtime_scale = 0.0010000000474974513  # Realistic D455 reported scale (~0.001 m/unit)

    # Conversion happens in sensor adapter:
    metric_depth = raw_depth.astype(np.float32) * runtime_scale

    assert metric_depth[0, 0] == pytest.approx(1.0, rel=1e-4)
    assert metric_depth[0, 1] == pytest.approx(2.5, rel=1e-4)
    assert metric_depth[1, 0] == pytest.approx(5.0, rel=1e-4)
    assert metric_depth[1, 1] == 0.0
    assert metric_depth.dtype == np.float32


def test_tum_depth_scale_conversion():
    """Verify TUM conversion: raw uint16 / 5000.0 = meters."""
    raw_depth = np.array([[5000, 10000], [2500, 0]], dtype=np.uint16)
    tum_scale = 5000.0

    metric_depth = raw_depth.astype(np.float32) / tum_scale

    assert metric_depth[0, 0] == pytest.approx(1.0, rel=1e-5)
    assert metric_depth[0, 1] == pytest.approx(2.0, rel=1e-5)
    assert metric_depth[1, 0] == pytest.approx(0.5, rel=1e-5)
    assert metric_depth[1, 1] == 0.0
    assert metric_depth.dtype == np.float32


def test_depth_converted_once_in_sensor_frame():
    """Canonical Stage 1 depth must be float32 meters; duplicate scale conversions are prevented."""
    metric_depth = np.full((480, 640), 2.5, dtype=np.float32)
    intrinsics = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)
    ts = Timestamp(value=1.0, domain=TimestampDomain.HARDWARE_CLOCK)

    frame = SensorFrame(
        session_id="session_test",
        sequence_number=0,
        timestamp=ts,
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        camera_intrinsics=intrinsics,
        depth=metric_depth,
        depth_scale=0.001,
        status=StreamStatus.OK,
    )

    # Invariant: depth remains float32 meters in the canonical frame
    assert frame.depth.dtype == np.float32
    assert frame.depth[0, 0] == pytest.approx(2.5)


def test_invalid_depth_scale_rejected():
    intrinsics = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)
    ts = Timestamp(value=1.0, domain=TimestampDomain.HARDWARE_CLOCK)

    with pytest.raises(ValueError, match="depth_scale must be positive and finite"):
        SensorFrame(
            session_id="session_test",
            sequence_number=0,
            timestamp=ts,
            rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            camera_intrinsics=intrinsics,
            depth_scale=0.0,
        )

    with pytest.raises(ValueError, match="depth_scale must be positive and finite"):
        SensorFrame(
            session_id="session_test",
            sequence_number=0,
            timestamp=ts,
            rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            camera_intrinsics=intrinsics,
            depth_scale=-0.001,
        )


def test_runtime_depth_scale_propagation():
    """Verify that runtime depth scale (meters/unit) propagates to SensorFrame and produces metric float32."""
    try:
        from ros2_ws.src.scene_graph_ros.scene_graph_ros.ros_conversions import (
            HAS_ROS2_MSGS,
            intrinsics_to_camera_info,
            numpy_to_ros_image,
            ros_messages_to_sensor_frame,
        )
    except ImportError:
        pytest.skip("ROS 2 conversions not importable in environment")

    if not HAS_ROS2_MSGS:
        pytest.skip("sensor_msgs not available in environment")

    rgb_arr = np.full((480, 640, 3), 100, dtype=np.uint8)
    depth_arr = np.full((480, 640), 2500, dtype=np.uint16)

    intr = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)
    rgb_msg = numpy_to_ros_image(rgb_arr, encoding="rgb8", frame_id="camera_color_frame", timestamp=1.0)
    depth_msg = numpy_to_ros_image(depth_arr, encoding="16uc1", frame_id="camera_color_frame", timestamp=1.0)
    cam_info = intrinsics_to_camera_info(intr, frame_id="camera_color_frame", timestamp=1.0)

    frame_d455 = ros_messages_to_sensor_frame(
        rgb_msg=rgb_msg,
        depth_msg=depth_msg,
        camera_info_msg=cam_info,
        session_id="test_d455",
        depth_scale=0.001,
    )
    assert frame_d455.depth_scale == pytest.approx(0.001)
    assert frame_d455.depth is not None
    assert frame_d455.depth.dtype == np.float32
    assert frame_d455.depth[0, 0] == pytest.approx(2.5)

    frame_calib = ros_messages_to_sensor_frame(
        rgb_msg=rgb_msg,
        depth_msg=depth_msg,
        camera_info_msg=cam_info,
        session_id="test_calib",
        depth_scale=0.001005,
    )
    assert frame_calib.depth_scale == pytest.approx(0.001005)
    assert frame_calib.depth[0, 0] == pytest.approx(2500 * 0.001005)

    frame_tum = ros_messages_to_sensor_frame(
        rgb_msg=rgb_msg,
        depth_msg=depth_msg,
        camera_info_msg=cam_info,
        session_id="test_tum",
        depth_scale=5000.0,
    )
    assert frame_tum.depth_scale == pytest.approx(0.0002)
    assert frame_tum.depth[0, 0] == pytest.approx(0.5)
