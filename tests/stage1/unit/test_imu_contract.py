"""Unit tests for Stage 1 IMU Data Contract and Bounded Buffering."""

from collections import deque
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from scene_graph.data.sensor_frame import IMUSample
from scene_graph.data.timestamp import TimestampDomain


def test_imu_sample_creation_and_immutability():
    sample = IMUSample(
        timestamp=100.123456,
        accel=np.array([0.1, 9.81, -0.2]),
        gyro=np.array([0.01, -0.02, 0.05]),
        domain=TimestampDomain.HARDWARE_CLOCK,
        sequence=1042,
    )

    assert sample.timestamp == pytest.approx(100.123456)
    assert np.allclose(sample.accel, [0.1, 9.81, -0.2])
    assert np.allclose(sample.gyro, [0.01, -0.02, 0.05])
    assert sample.domain == TimestampDomain.HARDWARE_CLOCK
    assert sample.sequence == 1042

    with pytest.raises(FrozenInstanceError):
        sample.timestamp = 200.0  # Frozen dataclass



def test_imu_bounded_ring_buffer():
    """Verify that a deque with maxlen=500 remains strictly bounded without unbounded growth."""
    imu_buffer: deque[IMUSample] = deque(maxlen=500)

    # Push 1000 high-rate samples (200 Hz for 5 seconds)
    for i in range(1000):
        sample = IMUSample(
            timestamp=float(i) * 0.005,
            accel=np.array([0.0, 9.81, 0.0]),
            gyro=np.array([0.0, 0.0, 0.0]),
            sequence=i,
        )
        imu_buffer.append(sample)

    assert len(imu_buffer) == 500
    # Oldest retained sample should be sequence 500 (earlier samples dropped O(1))
    assert imu_buffer[0].sequence == 500
    assert imu_buffer[-1].sequence == 999
