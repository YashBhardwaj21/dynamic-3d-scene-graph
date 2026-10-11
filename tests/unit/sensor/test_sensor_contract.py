"""Unit tests for Stage 1 Canonical SensorFrame Contract and Invariants."""

import numpy as np
import pytest

from scene_graph.data.frame_packet import FramePacket, LocalizationMode
from scene_graph.data.sensor_frame import IMUSample, SensorFrame, StreamStatus
from scene_graph.data.timestamp import Timestamp, TimestampDomain
from scene_graph.geometry.camera import CameraIntrinsics


def test_sensor_frame_invariants_valid():
    """Verify that a valid SensorFrame satisfies all Stage 1 invariants."""
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.float32) * 1.5
    intrinsics = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)
    ts = Timestamp(value=10.5, domain=TimestampDomain.HARDWARE_CLOCK, source="d455_color")

    frame = SensorFrame(
        session_id="session_alpha",
        sequence_number=1,
        timestamp=ts,
        rgb=rgb,
        camera_intrinsics=intrinsics,
        depth=depth,
        depth_scale=0.001,
        status=StreamStatus.OK,
    )

    assert frame.session_id == "session_alpha"
    assert frame.sequence_number == 1
    assert frame.timestamp.value == 10.5
    assert frame.has_depth is True
    assert frame.depth.dtype == np.float32
    assert frame.width == 640
    assert frame.height == 480
    assert frame.status == StreamStatus.OK


def test_sensor_frame_rejects_integer_depth():
    """Invariant 1: Canonical depth is float32 meters. Integer raw depth must be rejected."""
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_uint16 = np.ones((480, 640), dtype=np.uint16) * 1500
    intrinsics = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)
    ts = Timestamp(value=10.5, domain=TimestampDomain.HARDWARE_CLOCK)

    with pytest.raises(ValueError, match="must be metric float32 depth in meters"):
        SensorFrame(
            session_id="session_alpha",
            sequence_number=1,
            timestamp=ts,
            rgb=rgb,
            camera_intrinsics=intrinsics,
            depth=depth_uint16,
        )


def test_sensor_frame_rejects_invalid_sequence_and_session():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    intrinsics = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)
    ts = Timestamp(value=10.5, domain=TimestampDomain.HARDWARE_CLOCK)

    with pytest.raises(ValueError, match="sequence_number must be >= 0"):
        SensorFrame(
            session_id="session_alpha",
            sequence_number=-1,
            timestamp=ts,
            rgb=rgb,
            camera_intrinsics=intrinsics,
        )

    with pytest.raises(ValueError, match="session_id must be a non-empty string"):
        SensorFrame(
            session_id="",
            sequence_number=0,
            timestamp=ts,
            rgb=rgb,
            camera_intrinsics=intrinsics,
        )


def test_sensor_frame_to_frame_packet_adapter():
    """Verify that from_sensor_frame correctly bridges to downstream FramePacket without data loss."""
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.float32) * 2.0
    intrinsics = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)
    ts = Timestamp(value=12.345, domain=TimestampDomain.HARDWARE_CLOCK, source="d455")
    imu = (IMUSample(timestamp=12.344, accel=np.array([0.0, 9.8, 0.0]), gyro=np.array([0.0, 0.0, 0.0])),)

    sensor_frame = SensorFrame(
        session_id="session_beta",
        sequence_number=42,
        timestamp=ts,
        rgb=rgb,
        camera_intrinsics=intrinsics,
        depth=depth,
        depth_scale=0.001,
        imu_samples=imu,
        status=StreamStatus.OK,
    )

    pose = np.eye(4, dtype=np.float64)
    packet = FramePacket.from_sensor_frame(
        sensor_frame=sensor_frame,
        world_T_camera=pose,
        pose_timestamp=12.345,
        pose_source="slam",
        transform_source="slam_tf",
        transform_valid=True,
        localization_mode=LocalizationMode.WORLD_MODE,
    )

    assert packet.frame_index == 42
    assert packet.timestamp == 12.345
    assert packet.sensor_timestamp == 12.345
    assert packet.has_depth is True
    assert packet.depth[0, 0] == 2.0
    assert packet.has_pose is True
    assert packet.metadata["session_id"] == "session_beta"
    assert packet.metadata["stream_status"] == "ok"
    assert packet.metadata["timestamp_domain"] == "hardware_clock"
    assert len(packet.imu_samples) == 1


def test_imu_sample_creation_and_immutability():
    from dataclasses import FrozenInstanceError
    sample = IMUSample(
        timestamp=100.123456,
        accel=np.array([0.1, 9.81, -0.2]),
        gyro=np.array([0.01, -0.02, 0.05]),
        domain=TimestampDomain.HARDWARE_CLOCK,
        sequence=1042,
    )

    assert sample.timestamp == pytest.approx(100.123456)
    assert np.allclose(sample.accel, [0.1, 9.81, -0.2])
    assert np.allclose(sample.gyro, [0.01, -0.02, 0.05])
    assert sample.domain == TimestampDomain.HARDWARE_CLOCK
    assert sample.sequence == 1042

    with pytest.raises(FrozenInstanceError):
        sample.timestamp = 200.0


def test_imu_bounded_ring_buffer():
    from collections import deque
    imu_buffer: deque[IMUSample] = deque(maxlen=500)

    for i in range(1000):
        sample = IMUSample(
            timestamp=float(i) * 0.005,
            accel=np.array([0.0, 9.81, 0.0]),
            gyro=np.array([0.0, 0.0, 0.0]),
            sequence=i,
        )
        imu_buffer.append(sample)

    assert len(imu_buffer) == 500
    assert imu_buffer[0].sequence == 500
    assert imu_buffer[-1].sequence == 999


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
    try:
        from ros2_ws.src.scene_graph_ros.scene_graph_ros.ros_conversions import (
            HAS_ROS2_MSGS,
            camera_info_to_intrinsics,
            intrinsics_to_camera_info,
        )
    except ImportError:
        pytest.skip("ROS 2 conversions not importable in environment")

    if not HAS_ROS2_MSGS:
        pytest.skip("sensor_msgs not available in environment")

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
    assert msg.width == 640
    assert msg.height == 480
    assert msg.distortion_model == "plumb_bob"
    assert list(msg.d) == [-0.054, 0.062, 0.001, -0.002, 0.0]

    recovered = camera_info_to_intrinsics(msg)
    assert recovered.fx == pytest.approx(385.5)
    assert recovered.fy == pytest.approx(386.2)
    assert recovered.cx == pytest.approx(321.0)
    assert recovered.cy == pytest.approx(241.5)
    assert recovered.width == 640
    assert recovered.height == 480
