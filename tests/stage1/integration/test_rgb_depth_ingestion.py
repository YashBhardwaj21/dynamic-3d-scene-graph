"""Integration test: Ingesting RGB-D packets into canonical SensorFrame."""

import numpy as np

from scene_graph.data.sensor_frame import SensorFrame, StreamStatus
from scene_graph.data.timestamp import Timestamp, TimestampDomain
from scene_graph.geometry.camera import CameraIntrinsics


def test_rgb_depth_ingestion_parsing():
    raw_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    raw_depth = np.random.randint(500, 4000, (480, 640), dtype=np.uint16)

    runtime_scale = 0.001
    metric_depth = raw_depth.astype(np.float32) * runtime_scale

    intrinsics = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)
    ts = Timestamp(value=100.0, domain=TimestampDomain.HARDWARE_CLOCK)

    frame = SensorFrame(
        session_id="ingest_test",
        sequence_number=1,
        timestamp=ts,
        rgb=raw_rgb,
        camera_intrinsics=intrinsics,
        depth=metric_depth,
        depth_scale=runtime_scale,
        status=StreamStatus.OK,
    )

    assert frame.width == 640
    assert frame.height == 480
    assert frame.depth.dtype == np.float32
    assert np.all(frame.depth >= 0.5)
    assert np.all(frame.depth <= 4.0)
