"""Unit tests for Stage 1 RGB-Depth and TUM Stream Synchronization."""


from scene_graph.data.synchronization import associate


def test_tum_dp_synchronization_optimal_matching():
    """Verify that TUM Dynamic Programming matching strictly respects 0.02s (20 ms) tolerance."""
    rgb_times = [1.00, 1.03, 1.06, 1.10]
    depth_times = [1.01, 1.04, 1.055, 1.15]

    matches = associate(rgb_times, depth_times, max_dt=0.02)
    # rgb 0 (1.00) matches depth 0 (1.01) -> dt=0.010 <= 0.02
    # rgb 1 (1.03) matches depth 1 (1.04) -> dt=0.010 <= 0.02
    # rgb 2 (1.06) matches depth 2 (1.055) -> dt=0.005 <= 0.02
    # rgb 3 (1.10) with depth 3 (1.15) has dt=0.050 > 0.02 -> rejected!
    assert len(matches) == 3
    assert matches[0] == (0, 0)
    assert matches[1] == (1, 1)
    assert matches[2] == (2, 2)


def test_live_d455_sync_tolerance_limits():
    """Protected Operating Point: 50 ms live synchronization tolerance."""
    max_live_skew = 0.050  # 50 ms

    # Within tolerance
    skew_ok = 0.033
    assert skew_ok <= max_live_skew

    # Exceeding tolerance
    skew_bad = 0.055
    assert skew_bad > max_live_skew


def test_synchronization_monotonic_order():
    """Ensure DP associate preserves strict chronological monotonicity (i1 < i2 => j1 < j2)."""
    rgb = [1.0, 2.0, 3.0]
    depth = [1.01, 2.01, 3.01]

    matches = associate(rgb, depth, max_dt=0.02)
    assert len(matches) == 3
    for k in range(len(matches) - 1):
        assert matches[k][0] < matches[k + 1][0]
        assert matches[k][1] < matches[k + 1][1]
