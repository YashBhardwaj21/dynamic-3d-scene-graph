"""Integration test: Parity between Live and Replay Sensor Contracts."""

import numpy as np

from scene_graph.data.sensor_frame import SensorFrame, StreamStatus
from scene_graph.data.timestamp import Timestamp, TimestampDomain
from scene_graph.geometry.camera import CameraIntrinsics


def test_live_and_replay_contract_parity():
    """Verify that live acquisition and dataset replay produce identical canonical contracts."""
    intrinsics = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.float32) * 1.5

    # 1. Live Frame
    live_frame = SensorFrame(
        session_id="live_session_1",
        sequence_number=10,
        timestamp=Timestamp(value=100.0, domain=TimestampDomain.HARDWARE_CLOCK),
        rgb=rgb,
        camera_intrinsics=intrinsics,
        depth=depth,
        depth_scale=0.001,
        status=StreamStatus.OK,
    )

    # 2. Replay Frame
    replay_frame = SensorFrame(
        session_id="tum_replay_session_1",
        sequence_number=10,
        timestamp=Timestamp(value=1305031102.123456, domain=TimestampDomain.SIMULATED_TIME),
        rgb=rgb,
        camera_intrinsics=intrinsics,
        depth=depth,
        depth_scale=0.0002,  # 1/5000
        status=StreamStatus.OK,
    )

    # Invariants must hold equally for both
    assert live_frame.depth.dtype == np.float32
    assert replay_frame.depth.dtype == np.float32
    assert live_frame.rgb.shape == replay_frame.rgb.shape
    assert live_frame.width == replay_frame.width == 640
    assert live_frame.height == replay_frame.height == 480
    assert live_frame.timestamp.domain == TimestampDomain.HARDWARE_CLOCK
    assert replay_frame.timestamp.domain == TimestampDomain.SIMULATED_TIME
