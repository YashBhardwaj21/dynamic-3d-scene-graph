"""Integration test: High-rate IMU ingestion into independent ring buffer."""

from collections import deque

import numpy as np

from scene_graph.data.sensor_frame import IMUSample
from scene_graph.data.timestamp import TimestampDomain


def test_high_rate_imu_stream_buffering():
    """Verify that a 200 Hz IMU stream accumulates into a bounded history and can be windowed."""
    buffer: deque[IMUSample] = deque(maxlen=500)

    # 1. Simulate 200 Hz samples arriving over 0.5s (100 samples)
    t_start = 10.0
    for i in range(100):
        t_sample = t_start + i * 0.005
        sample = IMUSample(
            timestamp=t_sample,
            accel=np.array([0.01 * i, 9.81, -0.01 * i]),
            gyro=np.array([0.001 * i, 0.0, -0.001 * i]),
            domain=TimestampDomain.HARDWARE_CLOCK,
            sequence=i,
        )
        buffer.append(sample)

    assert len(buffer) == 100

    # 2. Windowed extraction between frame k-1 (10.1s) and frame k (10.3s)
    t_k_minus_1 = 10.1
    t_k = 10.3
    windowed = tuple(s for s in buffer if t_k_minus_1 < s.timestamp <= t_k)

    # In interval (10.1, 10.3], duration is 0.2s at 200 Hz = 40 samples
    assert len(windowed) == 40
    assert windowed[0].timestamp > t_k_minus_1
    assert windowed[-1].timestamp <= t_k
