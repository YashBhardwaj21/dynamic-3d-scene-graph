"""Unit tests for Stage 1 ClockMapping Offset and Drift Tracking."""

import pytest

from scene_graph.data.timestamp import ClockMapping


def test_clock_mapping_initialization_and_offset_recovery():
    mapping = ClockMapping(smoothing_alpha=0.1, max_drift_jump_sec=0.5, max_stale_duration_sec=2.0)
    assert not mapping.is_healthy
    assert mapping.offset is None

    # Sensor time: 10.0, WSL2 target time: 100.916 (simulating our observed 0.916s offset)
    healthy = mapping.update(source_time=10.0, target_time=100.916)
    assert healthy is True
    assert mapping.is_healthy is True
    assert mapping.offset == pytest.approx(90.916, rel=1e-4)

    # Project sensor time into target domain
    mapped = mapping.map_timestamp(10.5)
    assert mapped == pytest.approx(10.5 + 90.916, rel=1e-4)


def test_clock_mapping_rejects_sudden_jump():
    mapping = ClockMapping(smoothing_alpha=0.1, max_drift_jump_sec=0.5, max_stale_duration_sec=2.0)
    mapping.update(source_time=10.0, target_time=20.0)  # offset = 10.0
    assert mapping.is_healthy

    # Sudden jump of 2.0s (e.g. system clock change or camera reset)
    healthy = mapping.update(source_time=11.0, target_time=23.0)  # offset = 12.0 (jump = 2.0 > 0.5)
    assert healthy is False
    assert not mapping.is_healthy


def test_clock_mapping_reset():
    mapping = ClockMapping()
    mapping.update(source_time=5.0, target_time=15.0)
    assert mapping.is_healthy

    mapping.reset()
    assert not mapping.is_healthy
    assert mapping.offset is None
    assert mapping.sample_count == 0
