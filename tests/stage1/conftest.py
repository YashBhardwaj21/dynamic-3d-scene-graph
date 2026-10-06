"""Pytest configuration and shared fixtures for Stage 1 tests."""

import pytest


@pytest.fixture(autouse=True)
def ensure_ros2_context():
    """Ensure rclpy is initialized for any test creating ROS 2 nodes."""
    try:
        import rclpy
        if not rclpy.ok():
            rclpy.init()
        yield
    except ImportError:
        yield
