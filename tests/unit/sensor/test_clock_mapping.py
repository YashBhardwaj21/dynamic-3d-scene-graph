"""Unit tests for Stage 1 ClockMapping Offset, Drift, and Jitter Rejection."""

import pytest

from scene_graph.data.timestamp import ClockMapping


def test_clock_mapping_arbitrary_offsets():
    """Verify offset recovery across various realistic cross-OS time offsets."""
    for expected_offset in [0.0, 0.916, 125.45, -42.8, 1700000000.0]:
        mapping = ClockMapping(smoothing_alpha=0.2, max_drift_jump_sec=1.0)
        source_t = 10.0
        target_t = source_t + expected_offset

        healthy = mapping.update(source_time=source_t, target_time=target_t)
        assert healthy is True
        assert mapping.is_healthy is True
        assert mapping.offset == pytest.approx(expected_offset, abs=1e-5)

        # Verify mapping projection
        assert mapping.map_timestamp(source_t + 1.5) == pytest.approx(source_t + 1.5 + expected_offset, abs=1e-5)


def test_clock_mapping_rejects_network_transit_jitter():
    """Verify minimum-delay tracking filter rejects network transit spikes without offset distortion."""
    baseline_offset = 50.0  # True clock offset
    baseline_delay = 0.001  # 1 ms clean transport delay

    mapping = ClockMapping(smoothing_alpha=0.1, max_drift_jump_sec=1.0, window_size=20)

    # Send 30 samples at 30 FPS: most samples have baseline 1 ms delay,
    # but occasional samples suffer 50 ms network transit spikes
    for i in range(30):
        source_t = 10.0 + i * 0.0333
        if i in (5, 12, 19, 25):
            # Transit latency spike (50 ms network delay)
            delay = 0.050
        else:
            delay = baseline_delay

        target_t = source_t + baseline_offset + delay
        mapping.update(source_time=source_t, target_time=target_t)

    # Invariant: the estimated offset must reflect the minimal transit delay (~50.001),
    # completely rejecting the 50 ms transit spikes!
    expected_true_offset = baseline_offset + baseline_delay
    assert mapping.offset == pytest.approx(expected_true_offset, abs=0.005)
    assert mapping.is_healthy is True


def test_clock_mapping_tracks_physical_drift():
    """Verify that slow linear oscillator drift (e.g. 50 ppm) is smoothly tracked."""
    base_offset = 10.0
    drift_rate_ppm = 50e-6  # 50 microseconds per second

    mapping = ClockMapping(smoothing_alpha=0.1, max_drift_jump_sec=0.5, window_size=10)

    # Run 60 frames (2 seconds) with cumulative drift
    for i in range(60):
        source_t = 1.0 + i * 0.0333
        drift = i * 0.0333 * drift_rate_ppm
        target_t = source_t + base_offset + drift
        healthy = mapping.update(source_time=source_t, target_time=target_t)
        assert healthy is True

    assert mapping.is_healthy is True
    # Final offset reflects accumulated drift
    assert mapping.offset == pytest.approx(base_offset + 60 * 0.0333 * drift_rate_ppm, abs=1e-4)


def test_clock_mapping_monotonicity_guarantee():
    """Verify that mapped timestamps are strictly monotonically non-decreasing."""
    mapping = ClockMapping(smoothing_alpha=0.5, max_drift_jump_sec=1.0)
    mapping.update(source_time=10.0, target_time=20.0)  # offset = 10.0

    t1 = mapping.map_timestamp(10.0)  # 20.0
    t2 = mapping.map_timestamp(10.033)  # 20.033
    assert t2 >= t1

    # If offset slightly adjusts downward due to network jitter resolution
    mapping.update(source_time=10.066, target_time=20.050)
    t3 = mapping.map_timestamp(10.066)
    assert t3 >= t2  # Invariant: monotonic guarantee prevents backward jump!


def test_clock_mapping_discontinuity_and_reset():
    """Verify that large step jumps flag health warning and reset clears state."""
    mapping = ClockMapping(max_drift_jump_sec=0.5)
    mapping.update(source_time=10.0, target_time=20.0)  # offset = 10.0
    assert mapping.is_healthy

    # Sudden jump of 2.0s (e.g. NTP step or camera reboot)
    accepted = mapping.update(source_time=11.0, target_time=23.0)  # offset = 12.0
    assert accepted is False
    assert not mapping.is_healthy

    # Reset clears all state upon session change
    mapping.reset()
    assert not mapping.is_healthy
    assert mapping.offset is None
    assert mapping.sample_count == 0
