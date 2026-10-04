"""Stage 1 test fixtures package."""
from tests.stage1.fixtures.synthetic_generator import (
    make_synthetic_intrinsics,
    make_synthetic_packet_bytes,
    make_synthetic_rgbd,
    make_synthetic_sensor_frame,
)

__all__ = [
    "make_synthetic_intrinsics",
    "make_synthetic_packet_bytes",
    "make_synthetic_rgbd",
    "make_synthetic_sensor_frame",
]
