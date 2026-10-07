"""Integration test: ROS 2 message conversion to canonical SensorFrame."""

import numpy as np
import pytest

from ros2_ws.src.scene_graph_ros.scene_graph_ros.ros_conversions import (
    HAS_ROS2_MSGS,
    intrinsics_to_camera_info,
    numpy_to_ros_image,
    ros_messages_to_sensor_frame,
)
from scene_graph.geometry.camera import CameraIntrinsics


def test_ros_messages_to_sensor_frame():
    if not HAS_ROS2_MSGS:
        pytest.skip("sensor_msgs not available in environment")
    rgb_arr = np.full((480, 640, 3), 120, dtype=np.uint8)
    depth_arr = np.full((480, 640), 1500, dtype=np.uint16)

    intr = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)

    rgb_msg = numpy_to_ros_image(rgb_arr, encoding="rgb8", frame_id="camera_color_frame", timestamp=10.0)
    depth_msg = numpy_to_ros_image(depth_arr, encoding="16uc1", frame_id="camera_depth_frame", timestamp=10.0)
    cam_info = intrinsics_to_camera_info(intr, frame_id="camera_color_frame", timestamp=10.0)

    sensor_frame = ros_messages_to_sensor_frame(
        rgb_msg=rgb_msg,
        depth_msg=depth_msg,
        camera_info_msg=cam_info,
        session_id="ros_test_session",
        sequence_number=1,
        depth_scale=1000.0,
    )

    assert sensor_frame.session_id == "ros_test_session"
    assert sensor_frame.sequence_number == 1
    assert sensor_frame.timestamp.value == pytest.approx(10.0)
    assert sensor_frame.has_depth is True
    assert sensor_frame.depth.dtype == np.float32
    assert sensor_frame.depth[0, 0] == pytest.approx(1.5)
    assert sensor_frame.camera_intrinsics.fx == pytest.approx(385.0)
