"""Synthetic Stage 1 frame and packet generator for deterministic testing and benchmarking."""

from __future__ import annotations

import json
import struct

import numpy as np

from scene_graph.data.sensor_frame import SensorFrame, StreamStatus
from scene_graph.data.timestamp import Timestamp, TimestampDomain
from scene_graph.geometry.camera import CameraIntrinsics


def make_synthetic_intrinsics(
    width: int = 640,
    height: int = 480,
    fx: float = 385.0,
    fy: float = 385.0,
    cx: float = 320.0,
    cy: float = 240.0,
) -> CameraIntrinsics:
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


def make_synthetic_rgbd(
    width: int = 640,
    height: int = 480,
    color_val: int = 128,
    depth_m: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.full((height, width, 3), color_val, dtype=np.uint8)
    depth = np.full((height, width), depth_m, dtype=np.float32)
    return rgb, depth


def make_synthetic_sensor_frame(
    sequence_number: int = 0,
    timestamp_sec: float = 100.0,
    session_id: str = "test_session_001",
    domain: TimestampDomain = TimestampDomain.HARDWARE_CLOCK,
    width: int = 640,
    height: int = 480,
    depth_m: float = 1.5,
) -> SensorFrame:
    rgb, depth = make_synthetic_rgbd(width=width, height=height, depth_m=depth_m)
    intrinsics = make_synthetic_intrinsics(width=width, height=height)
    ts = Timestamp(value=timestamp_sec, domain=domain, source="synthetic_d455")
    return SensorFrame(
        session_id=session_id,
        sequence_number=sequence_number,
        timestamp=ts,
        rgb=rgb,
        camera_intrinsics=intrinsics,
        depth=depth,
        depth_scale=0.001,
        status=StreamStatus.OK,
    )


def make_synthetic_packet_bytes(
    sequence_number: int = 0,
    timestamp_sec: float = 100.0,
    skew_ms: float = 0.5,
    width: int = 640,
    height: int = 480,
    protocol_version: int = 2,
    session_id: str = "test_session_001",
) -> bytes:
    color_bytes = np.full((height, width, 3), 128, dtype=np.uint8).tobytes()
    depth_bytes = np.full((height, width), 1500, dtype=np.uint16).tobytes()

    header = {
        "protocol_version": protocol_version,
        "type": "d455_rgbd",
        "session_id": session_id,
        "frame_id": sequence_number,
        "host_monotonic_time": timestamp_sec,
        "host_wall_time": timestamp_sec + 1000.0,
        "color": {
            "timestamp": timestamp_sec,
            "timestamp_domain": "hardware_clock",
            "width": width,
            "height": height,
            "format": "bgr8",
            "channels": 3,
            "bytes_per_channel": 1,
            "payload_size": len(color_bytes),
            "fx": 385.0,
            "fy": 385.0,
            "ppx": 320.0,
            "ppy": 240.0,
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "distortion_model": "plumb_bob",
        },
        "depth": {
            "timestamp": timestamp_sec + (skew_ms * 1e-3),
            "timestamp_domain": "hardware_clock",
            "width": width,
            "height": height,
            "format": "z16",
            "channels": 1,
            "bytes_per_channel": 2,
            "payload_size": len(depth_bytes),
            "depth_scale": 0.001,
            "is_aligned_to_color": True,
        },
        "imu": [
            {
                "timestamp": timestamp_sec,
                "domain": "hardware_clock",
                "accel": [0.0, 9.81, 0.0],
                "gyro": [0.01, 0.0, -0.01],
            }
        ],
        "rgb_depth_dt_ms": skew_ms,
    }

    header_bytes = json.dumps(header).encode("utf-8")
    return struct.pack("!I", len(header_bytes)) + header_bytes + color_bytes + depth_bytes
