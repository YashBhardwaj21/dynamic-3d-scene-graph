"""Unit tests for Timestamp Container, Validation, Provenance, and Arithmetic."""

import pytest

from scene_graph.data.timestamp import (
    IncompatibleTimestampDomainError,
    Timestamp,
    TimestampDomain,
)


def test_timestamp_valid_creation():
    ts = Timestamp(value=100.5, domain=TimestampDomain.HARDWARE_CLOCK, source="d455")
    assert ts.value == 100.5
    assert ts.domain == TimestampDomain.HARDWARE_CLOCK
    assert ts.source == "d455"


def test_timestamp_rejects_negative_or_infinite():
    with pytest.raises(ValueError, match="must be non-negative and finite"):
        Timestamp(value=-1.0, domain=TimestampDomain.HARDWARE_CLOCK)

    with pytest.raises(ValueError, match="must be non-negative and finite"):
        Timestamp(value=float("inf"), domain=TimestampDomain.HARDWARE_CLOCK)

    with pytest.raises(ValueError, match="must be non-negative and finite"):
        Timestamp(value=float("nan"), domain=TimestampDomain.HARDWARE_CLOCK)


def test_timestamp_comparisons_same_domain_and_source():
    t1 = Timestamp(value=10.0, domain=TimestampDomain.HARDWARE_CLOCK, source="d455")
    t2 = Timestamp(value=15.0, domain=TimestampDomain.HARDWARE_CLOCK, source="d455")

    assert t1.is_comparable_with(t2)
    assert t1 < t2
    assert t1 <= t2
    assert t2 > t1
    assert t2 >= t1
    assert t1.delta_to(t2) == pytest.approx(5.0)


def test_timestamp_comparisons_cross_source_incompatible():
    """Verify that same-domain timestamps from distinct uncalibrated sources reject direct comparison."""
    t_win = Timestamp(value=100.0, domain=TimestampDomain.SYSTEM_TIME, source="windows_host")
    t_wsl = Timestamp(value=100.916, domain=TimestampDomain.SYSTEM_TIME, source="wsl2_ros")

    # Invariant: cross-OS clocks have unmodeled offsets and cannot be silently compared
    assert not t_win.is_comparable_with(t_wsl)

    with pytest.raises(IncompatibleTimestampDomainError, match="Cannot compare timestamp"):
        _ = t_win.delta_to(t_wsl)

    with pytest.raises(IncompatibleTimestampDomainError, match="Cannot order timestamp"):
        _ = t_win < t_wsl


def test_timestamp_comparisons_cross_domain_incompatible():
    t_hw = Timestamp(value=10.0, domain=TimestampDomain.HARDWARE_CLOCK, source="d455_hw")
    t_sys = Timestamp(value=1700000000.0, domain=TimestampDomain.SYSTEM_TIME, source="windows_os")

    assert not t_hw.is_comparable_with(t_sys)

    with pytest.raises(IncompatibleTimestampDomainError, match="Cannot compare timestamp"):
        _ = t_hw.delta_to(t_sys)

    with pytest.raises(IncompatibleTimestampDomainError, match="Cannot order timestamp"):
        _ = t_hw < t_sys
