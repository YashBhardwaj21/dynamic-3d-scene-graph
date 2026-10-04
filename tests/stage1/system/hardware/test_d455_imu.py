# ruff: noqa: BLE001
"""Hardware test: D455 6-axis IMU streams (accel & gyro)."""

import pytest

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


def test_d455_hardware_imu():
    if rs is None:
        pytest.skip("pyrealsense2 is not installed. Hardware test skipped.")

    ctx = rs.context()
    if len(ctx.query_devices()) == 0:
        pytest.skip("No Intel RealSense device detected on USB. Hardware test skipped.")

    pipeline = rs.pipeline()
    config = rs.config()
    try:
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
        pipeline.start(config)
    except Exception as ex:
        pytest.skip(f"IMU stream configuration unavailable on this device: {ex}")

    try:
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        accel_f = frames.first_or_default(rs.stream.accel)
        gyro_f = frames.first_or_default(rs.stream.gyro)

        assert accel_f is not None
        assert gyro_f is not None
        accel_data = accel_f.as_motion_frame().get_motion_data()
        gyro_data = gyro_f.as_motion_frame().get_motion_data()

        # Gravity check: norm should be around 9.8 m/s^2 when stationary
        norm_accel = (accel_data.x**2 + accel_data.y**2 + accel_data.z**2)**0.5
        print(f"[Hardware IMU] Accel norm: {norm_accel:.2f} m/s^2 | Gyro: [{gyro_data.x:.2f}, {gyro_data.y:.2f}, {gyro_data.z:.2f}] rad/s")
        assert 7.0 < norm_accel < 13.0
    finally:
        pipeline.stop()
