"""Unit test: Session ID continuity, timing provenance, and hardware skew telemetry."""

import numpy as np
import pytest

from ros2_ws.src.scene_graph_ros.scene_graph_ros.ros_conversions import (
    HAS_ROS2_MSGS,
    intrinsics_to_camera_info,
    numpy_to_ros_image,
    ros_messages_to_sensor_frame,
)
from scene_graph.data.timestamp import Timestamp, TimestampDomain
from scene_graph.geometry.camera import CameraIntrinsics


def test_session_id_propagation():
    """Verify that runtime session_id propagates to SensorFrame without hardcoded overrides."""
    if not HAS_ROS2_MSGS:
        pytest.skip("sensor_msgs not available in environment")

    rgb_arr = np.full((480, 640, 3), 100, dtype=np.uint8)
    depth_arr = np.full((480, 640), 2000, dtype=np.uint16)
    intr = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)

    rgb_msg = numpy_to_ros_image(rgb_arr, encoding="rgb8", frame_id="camera_color_frame", timestamp=10.0)
    depth_msg = numpy_to_ros_image(depth_arr, encoding="16uc1", frame_id="camera_color_frame", timestamp=10.0)
    cam_info = intrinsics_to_camera_info(intr, frame_id="camera_color_frame", timestamp=10.0)

    unique_session = "d455_unique_run_9999"
    sensor_frame = ros_messages_to_sensor_frame(
        rgb_msg=rgb_msg,
        depth_msg=depth_msg,
        camera_info_msg=cam_info,
        session_id=unique_session,
        sequence_number=42,
    )

    assert sensor_frame.session_id == unique_session
    assert sensor_frame.session_id != "ros2_live_session"
    assert sensor_frame.sequence_number == 42


def test_timing_provenance_preservation():
    """Verify that host capture, network arrival, and mapped ROS timestamps are preserved."""
    if not HAS_ROS2_MSGS:
        pytest.skip("sensor_msgs not available in environment")

    rgb_arr = np.full((480, 640, 3), 100, dtype=np.uint8)
    depth_arr = np.full((480, 640), 2000, dtype=np.uint16)
    intr = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)

    rgb_msg = numpy_to_ros_image(rgb_arr, encoding="rgb8", frame_id="camera_color_frame", timestamp=100.05)
    depth_msg = numpy_to_ros_image(depth_arr, encoding="16uc1", frame_id="camera_color_frame", timestamp=100.05)
    cam_info = intrinsics_to_camera_info(intr, frame_id="camera_color_frame", timestamp=100.05)

    host_cap_time = 100.00
    net_arr_time = 100.02
    mapped_time = 100.05
    src_time = 12345.678  # hardware sensor clock

    sensor_frame = ros_messages_to_sensor_frame(
        rgb_msg=rgb_msg,
        depth_msg=depth_msg,
        camera_info_msg=cam_info,
        session_id="session_prov",
        sequence_number=10,
        host_capture_timestamp=host_cap_time,
        network_arrival_timestamp=net_arr_time,
        mapped_ros_timestamp=mapped_time,
        source_timestamp=src_time,
        source_domain=TimestampDomain.HARDWARE_CLOCK,
    )

    # Primary ROS/TF timestamp
    assert sensor_frame.timestamp.value == pytest.approx(mapped_time)
    # Timing provenance
    assert sensor_frame.host_capture_timestamp is not None
    assert sensor_frame.host_capture_timestamp.value == pytest.approx(host_cap_time)
    assert sensor_frame.host_capture_timestamp.domain == TimestampDomain.SYSTEM_TIME

    assert sensor_frame.network_arrival_timestamp is not None
    assert sensor_frame.network_arrival_timestamp.value == pytest.approx(net_arr_time)
    assert sensor_frame.network_arrival_timestamp.domain == TimestampDomain.SYSTEM_TIME

    assert sensor_frame.mapped_ros_timestamp == pytest.approx(mapped_time)

    # Provenance in metadata
    assert sensor_frame.metadata.get("source_timestamp") == pytest.approx(src_time)
    assert sensor_frame.metadata.get("source_domain") == "hardware_clock"


def test_hardware_skew_telemetry():
    """Verify that original hardware RGB-depth skew is preserved in metadata."""
    if not HAS_ROS2_MSGS:
        pytest.skip("sensor_msgs not available in environment")

    rgb_arr = np.full((480, 640, 3), 100, dtype=np.uint8)
    depth_arr = np.full((480, 640), 2000, dtype=np.uint16)
    intr = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)

    rgb_msg = numpy_to_ros_image(rgb_arr, encoding="rgb8", frame_id="camera_color_frame", timestamp=50.0)
    depth_msg = numpy_to_ros_image(depth_arr, encoding="16uc1", frame_id="camera_color_frame", timestamp=50.0)
    cam_info = intrinsics_to_camera_info(intr, frame_id="camera_color_frame", timestamp=50.0)

    raw_skew_ms = 4.25  # Real hardware skew measured at camera
    meta = {"rgb_depth_dt_ms": raw_skew_ms}

    sensor_frame = ros_messages_to_sensor_frame(
        rgb_msg=rgb_msg,
        depth_msg=depth_msg,
        camera_info_msg=cam_info,
        session_id="session_skew",
        metadata=meta,
    )

    # Even though rgb_msg and depth_msg have matching mapped timestamps (skew=0.0 on ROS level),
    # the original hardware skew is faithfully preserved in frame metadata
    assert sensor_frame.metadata["rgb_depth_dt_ms"] == pytest.approx(4.25)
