"""Unit test: Runtime D455 depth scale propagation and metric conversion."""

import numpy as np
import pytest

from ros2_ws.src.scene_graph_ros.scene_graph_ros.ros_conversions import (
    HAS_ROS2_MSGS,
    intrinsics_to_camera_info,
    numpy_to_ros_image,
    ros_messages_to_sensor_frame,
)
from scene_graph.data.sensor_frame import SensorFrame
from scene_graph.geometry.camera import CameraIntrinsics


def test_runtime_depth_scale_propagation():
    """Verify that runtime depth scale (meters/unit) propagates to SensorFrame and produces metric float32."""
    if not HAS_ROS2_MSGS:
        pytest.skip("sensor_msgs not available in environment")

    rgb_arr = np.full((480, 640, 3), 100, dtype=np.uint8)
    depth_arr = np.full((480, 640), 2500, dtype=np.uint16)  # 2500 raw units

    intr = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)
    rgb_msg = numpy_to_ros_image(rgb_arr, encoding="rgb8", frame_id="camera_color_frame", timestamp=1.0)
    depth_msg = numpy_to_ros_image(depth_arr, encoding="16uc1", frame_id="camera_color_frame", timestamp=1.0)
    cam_info = intrinsics_to_camera_info(intr, frame_id="camera_color_frame", timestamp=1.0)

    # 1. Standard D455 runtime scale: 0.001 m/unit (1 mm)
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
    # 2500 * 0.001 = 2.500 meters
    assert frame_d455.depth[0, 0] == pytest.approx(2.5)

    # 2. Calibrated D455 runtime scale: 0.001005 m/unit
    frame_calib = ros_messages_to_sensor_frame(
        rgb_msg=rgb_msg,
        depth_msg=depth_msg,
        camera_info_msg=cam_info,
        session_id="test_calib",
        depth_scale=0.001005,
    )
    assert frame_calib.depth_scale == pytest.approx(0.001005)
    # 2500 * 0.001005 = 2.5125 meters
    assert frame_calib.depth[0, 0] == pytest.approx(2500 * 0.001005)

    # 3. TUM dataset scale: 5000 raw units per meter (= 0.0002 m/unit)
    frame_tum = ros_messages_to_sensor_frame(
        rgb_msg=rgb_msg,
        depth_msg=depth_msg,
        camera_info_msg=cam_info,
        session_id="test_tum",
        depth_scale=5000.0,
    )
    assert frame_tum.depth_scale == pytest.approx(0.0002)
    # 2500 / 5000.0 = 0.500 meters
    assert frame_tum.depth[0, 0] == pytest.approx(0.5)

    # 4. Fallback legacy units-per-meter scale: 1000.0
    frame_legacy = ros_messages_to_sensor_frame(
        rgb_msg=rgb_msg,
        depth_msg=depth_msg,
        camera_info_msg=cam_info,
        session_id="test_legacy",
        depth_scale=1000.0,
    )
    assert frame_legacy.depth_scale == pytest.approx(0.001)
    assert frame_legacy.depth[0, 0] == pytest.approx(2.5)
