"""Hardware test: D455 factory calibration acquisition."""

import pytest

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


def test_d455_hardware_calibration():
    if rs is None:
        pytest.skip("pyrealsense2 is not installed. Hardware test skipped.")

    ctx = rs.context()
    if len(ctx.query_devices()) == 0:
        pytest.skip("No Intel RealSense device detected on USB. Hardware test skipped.")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        profile = pipeline.start(config)
        color_p = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_p.get_intrinsics()

        print(f"[Hardware Intrinsics] fx={intr.fx}, fy={intr.fy}, ppx={intr.ppx}, ppy={intr.ppy}")
        assert intr.fx > 100.0
        assert intr.fy > 100.0
        assert 0.0 < intr.ppx < 640.0
        assert 0.0 < intr.ppy < 480.0
    finally:
        pipeline.stop()
