"""Hardware test: D455 timestamp domains and frame counter monotonicity."""

import pytest

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


def test_d455_hardware_timestamps():
    if rs is None:
        pytest.skip("pyrealsense2 is not installed. Hardware test skipped.")

    ctx = rs.context()
    if len(ctx.query_devices()) == 0:
        pytest.skip("No Intel RealSense device detected on USB. Hardware test skipped.")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
        last_ts = None
        for _ in range(10):
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            c_frame = frames.get_color_frame()
            ts = c_frame.get_timestamp()
            domain = c_frame.get_frame_timestamp_domain()

            assert domain in (
                rs.timestamp_domain.hardware_clock,
                rs.timestamp_domain.system_time,
                rs.timestamp_domain.global_time,
            )
            if last_ts is not None:
                assert ts > last_ts
            last_ts = ts
    finally:
        pipeline.stop()
