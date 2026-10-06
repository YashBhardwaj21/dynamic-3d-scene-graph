"""Unit tests for Stage 1 Calibration Acquisition, Validation, and End-to-End Preservation."""

import numpy as np
import pytest

from ros2_ws.src.scene_graph_ros.scene_graph_ros.ros_conversions import (
    camera_info_to_intrinsics,
    intrinsics_to_camera_info,
)
from scene_graph.data.sensor_frame import SensorFrame
from scene_graph.data.timestamp import Timestamp, TimestampDomain
from scene_graph.geometry.camera import CameraIntrinsics


def test_calibration_intrinsics_preserved():
    intr = CameraIntrinsics(
        fx=385.123,
        fy=385.456,
        cx=321.789,
        cy=239.012,
        width=640,
        height=480,
    )
    distortion = (-0.05, 0.06, 0.001, -0.002, 0.0)
    ts = Timestamp(value=1.0, domain=TimestampDomain.HARDWARE_CLOCK)

    frame = SensorFrame(
        session_id="session_calib",
        sequence_number=1,
        timestamp=ts,
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        camera_intrinsics=intr,
        distortion=distortion,
        distortion_model="brown_conrady",
    )

    assert frame.camera_intrinsics.fx == pytest.approx(385.123)
    assert frame.camera_intrinsics.fy == pytest.approx(385.456)
    assert frame.camera_intrinsics.cx == pytest.approx(321.789)
    assert frame.camera_intrinsics.cy == pytest.approx(239.012)
    assert frame.distortion == distortion
    assert frame.distortion_model == "brown_conrady"


def test_calibration_resolution_validation():
    """Verify that resolution matches stream dimensions and catches dimension mismatches."""
    intr_correct = CameraIntrinsics(
        fx=385.0,
        fy=385.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )

    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    ts = Timestamp(value=1.0, domain=TimestampDomain.HARDWARE_CLOCK)

    frame = SensorFrame(
        session_id="session_res",
        sequence_number=1,
        timestamp=ts,
        rgb=rgb,
        camera_intrinsics=intr_correct,
    )
    assert frame.width == 640
    assert frame.height == 480

    # Inconsistent resolution: image is 480x640, but intrinsics claim 720x1280
    intr_mismatch = CameraIntrinsics(
        fx=700.0,
        fy=700.0,
        cx=640.0,
        cy=360.0,
        width=1280,
        height=720,
    )
    with pytest.raises(ValueError, match="Image shape .* does not match camera intrinsics"):
        SensorFrame(
            session_id="session_mismatch",
            sequence_number=2,
            timestamp=ts,
            rgb=rgb,
            camera_intrinsics=intr_mismatch,
        )


def test_calibration_end_to_end_ros_roundtrip():
    """Verify that CameraIntrinsics <-> ROS CameraInfo preserves K, D, P, R, and distortion model."""
    intr = CameraIntrinsics(
        fx=385.5,
        fy=386.2,
        cx=321.0,
        cy=241.5,
        width=640,
        height=480,
        distortion=(-0.054, 0.062, 0.001, -0.002, 0.0),
        distortion_model="plumb_bob",
    )

    msg = intrinsics_to_camera_info(intr, frame_id="camera_color_optical_frame", timestamp=1.5)

    # Verify ROS CameraInfo field structure
    assert msg.header.stamp.sec == 1
    assert msg.header.stamp.nanosec == 500_000_000
    assert msg.header.frame_id == "camera_color_optical_frame"
    assert msg.width == 640
    assert msg.height == 480
    assert msg.distortion_model == "plumb_bob"
    assert list(msg.d) == [-0.054, 0.062, 0.001, -0.002, 0.0]
    # K matrix
    assert msg.k[0] == 385.5
    assert msg.k[2] == 321.0
    assert msg.k[4] == 386.2
    assert msg.k[5] == 241.5
    assert msg.k[8] == 1.0
    # R is identity
    assert list(msg.r) == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    # P projection matrix
    assert msg.p[0] == 385.5
    assert msg.p[2] == 321.0
    assert msg.p[5] == 386.2
    assert msg.p[6] == 241.5

    # Roundtrip back to CameraIntrinsics
    recovered = camera_info_to_intrinsics(msg)
    assert recovered.fx == pytest.approx(385.5)
    assert recovered.fy == pytest.approx(386.2)
    assert recovered.cx == pytest.approx(321.0)
    assert recovered.cy == pytest.approx(241.5)
    assert recovered.width == 640
    assert recovered.height == 480
    assert recovered.distortion_model == "plumb_bob"
    assert list(recovered.distortion) == [-0.054, 0.062, 0.001, -0.002, 0.0]
