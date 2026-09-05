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

def test_associate_optimal_monotonic_matching():
    """Verify DP algorithm achieves maximum cardinality even when local NN would fail.
    
    Adversarial case:
    primary = [1.0, 1.05]
    secondary = [0.95, 1.0]
    max_dt = 0.1
    
    A local nearest-neighbor algorithm would greedily match P[0] (1.0) with S[1] (1.0) 
    because diff=0.0. Then P[1] (1.05) cannot match S[0] (0.95) because that violates 
    monotonic ordering. Total matches = 1.
    
    The optimal global DP algorithm maximizes cardinality by matching:
    P[0] -> S[0] (diff 0.05)
    P[1] -> S[1] (diff 0.05)
    Total matches = 2.
    """
    primary = [1.0, 1.05]
    secondary = [0.95, 1.0]
    
    matches = associate(primary, secondary, max_dt=0.1)
    
    assert len(matches) == 2
    assert matches == [(0, 0), (1, 1)]

def test_associate_asymmetric_lists():
    """Verify behavior with different length lists and missing data."""
    primary = [1.0, 2.0, 3.0, 4.0, 5.0]
    secondary = [2.01, 4.0]
    
    matches = associate(primary, secondary, max_dt=0.02)
    assert matches == [(1, 0), (3, 1)]
