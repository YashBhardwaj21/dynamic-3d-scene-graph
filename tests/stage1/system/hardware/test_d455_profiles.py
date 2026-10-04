"""Hardware test: D455 active stream profiles (640x480 @ 30 FPS BGR8 & Z16)."""

import pytest

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


def test_d455_hardware_stream_profiles():
    if rs is None:
        pytest.skip("pyrealsense2 is not installed. Hardware test skipped.")

    ctx = rs.context()
    if len(ctx.query_devices()) == 0:
        pytest.skip("No Intel RealSense device detected on USB. Hardware test skipped.")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        profile = pipeline.start(config)
        color_p = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_p = profile.get_stream(rs.stream.depth).as_video_stream_profile()

        assert color_p.width() == 640
        assert color_p.height() == 480
        assert color_p.fps() == 30

        assert depth_p.width() == 640
        assert depth_p.height() == 480
        assert depth_p.fps() == 30
    finally:
        pipeline.stop()
