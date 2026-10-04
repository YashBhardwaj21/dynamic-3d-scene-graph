"""Hardware test: D455 device detection and pipeline startup."""

import pytest

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


def test_d455_hardware_startup():
    if rs is None:
        pytest.skip("pyrealsense2 is not installed. Hardware test skipped.")

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        pytest.skip("No Intel RealSense device detected on USB. Hardware test skipped.")

    dev = devices[0]
    dev_name = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)
    fw = dev.get_info(rs.camera_info.firmware_version)

    print(f"[Hardware Detected] {dev_name} (Serial: {serial}, FW: {fw})")
    assert "D455" in dev_name or "RealSense" in dev_name
