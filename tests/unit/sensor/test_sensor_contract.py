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
