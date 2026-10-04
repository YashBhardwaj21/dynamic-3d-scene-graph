"""Hardware test: D455 runtime depth scale query."""

import pytest

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


def test_d455_hardware_depth_scale():
    if rs is None:
        pytest.skip("pyrealsense2 is not installed. Hardware test skipped.")

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        pytest.skip("No Intel RealSense device detected on USB. Hardware test skipped.")

    dev = devices[0]
    depth_sensor = dev.first_depth_sensor()
    scale = depth_sensor.get_depth_scale()

    print(f"[Hardware Depth Scale] {scale:.8f} m/unit")
    # RealSense D455 typically reports 0.001 m/unit (1 mm per raw count)
    assert 0.0001 < scale < 0.01
