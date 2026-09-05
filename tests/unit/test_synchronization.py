"""Unit tests for Timestamp synchronization (Substage 1.2)."""

from scene_graph.data.synchronization import associate

def test_associate_exact_match():
    """Verify exact matches are found correctly."""
    primary = [1.0, 2.0, 3.0]
    secondary = [1.0, 2.0, 3.0]
    
    matches = associate(primary, secondary, max_dt=0.02)
    assert matches == [(0, 0), (1, 1), (2, 2)]

def test_associate_within_threshold():
    """Verify matches within max_dt are found."""
    primary = [1.0, 2.0, 3.0]
    secondary = [1.01, 1.99, 3.015]
    
    matches = associate(primary, secondary, max_dt=0.02)
    assert matches == [(0, 0), (1, 1), (2, 2)]

def test_associate_exceeds_threshold():
    """Verify matches exceeding max_dt are rejected."""
    primary = [1.0, 2.0, 3.0]
    secondary = [1.03, 2.0, 3.03]  # 1.03 and 3.03 are > 0.02 away
    
    matches = associate(primary, secondary, max_dt=0.02)
    assert matches == [(1, 1)]

def test_associate_greedy_assignment():
    """Verify greedy assignment minimizes time difference and respects 1-to-1."""
    # p[0] is distance 0.01 from s[0] and 0.015 from s[1]
    # p[1] is distance 0.005 from s[1]
    # Greedy should assign (p[1], s[1]) first, then (p[0], s[0])
    primary = [1.0, 1.01]
    secondary = [0.99, 1.015]
    
    matches = associate(primary, secondary, max_dt=0.02)
    # diffs:
    # (p=0, s=0) -> |1.0 - 0.99| = 0.01
    # (p=0, s=1) -> |1.0 - 1.015| = 0.015
    # (p=1, s=0) -> |1.01 - 0.99| = 0.02
    # (p=1, s=1) -> |1.01 - 1.015| = 0.005
    # Smallest diff is (p=1, s=1) with 0.005
    # Next smallest is (p=0, s=0) with 0.01
    assert matches == [(0, 0), (1, 1)]

def test_associate_asymmetric_lists():
    """Verify behavior with different length lists and missing data."""
    primary = [1.0, 2.0, 3.0, 4.0, 5.0]
    secondary = [2.01, 4.0]
    
    matches = associate(primary, secondary, max_dt=0.02)
    assert matches == [(1, 0), (3, 1)]
