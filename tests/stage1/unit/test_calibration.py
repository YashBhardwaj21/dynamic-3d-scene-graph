"""Unit tests for Stage 1 Calibration Acquisition and Preservation."""

import pytest

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
    import numpy as np

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
    """Verify that resolution matches stream dimensions."""
    intr = CameraIntrinsics(
        fx=385.0,
        fy=385.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )
    import numpy as np

    # Image is 480x640, matching intrinsics
    rgb_valid = np.zeros((480, 640, 3), dtype=np.uint8)
    ts = Timestamp(value=1.0, domain=TimestampDomain.HARDWARE_CLOCK)

    frame = SensorFrame(
        session_id="session_res",
        sequence_number=1,
        timestamp=ts,
        rgb=rgb_valid,
        camera_intrinsics=intr,
    )
    assert frame.width == 640
    assert frame.height == 480
