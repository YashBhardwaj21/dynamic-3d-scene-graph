"""Timestamp synchronization for TUM RGB-D sequences.

Implements nearest-neighbour matching for timestamps across multiple sensors
(RGB, Depth, Ground Truth) with a strict maximum time difference (max_dt).
Ensures 1-to-1 matching by resolving conflicts greedily.
"""

from typing import List, Tuple


def associate(
    primary_timestamps: List[float], 
    secondary_timestamps: List[float], 
    max_dt: float = 0.02
) -> List[Tuple[int, int]]:
    """Greedy nearest-neighbour timestamp matching.

    Matches timestamps from a primary list to a secondary list.
    Ensures that each primary and secondary timestamp is used at most once.

    Args:
        primary_timestamps: List of timestamps (e.g., RGB).
        secondary_timestamps: List of timestamps (e.g., Depth).
        max_dt: Maximum allowed time difference in seconds.

    Returns:
        List of matched index pairs (primary_idx, secondary_idx) sorted by primary_idx.
    """
    # Create all potential matches within max_dt
    potential_matches = []
    
    # We use a simple O(N*M) loop which is extremely fast for N,M < 10000 in pure Python,
    # but we can optimize slightly by assuming timestamps are roughly sorted.
    # To keep it bulletproof and match standard TUM associate.py behavior exactly:
    for p_idx, p_t in enumerate(primary_timestamps):
        for s_idx, s_t in enumerate(secondary_timestamps):
            diff = abs(p_t - s_t)
            if diff <= max_dt:
                potential_matches.append((diff, p_idx, s_idx))
                
    # Sort by time difference (smallest first) for greedy assignment
    potential_matches.sort(key=lambda x: x[0])
    
    matches = []
    used_primary = set()
    used_secondary = set()
    
    for diff, p_idx, s_idx in potential_matches:
        if p_idx not in used_primary and s_idx not in used_secondary:
            matches.append((p_idx, s_idx))
            used_primary.add(p_idx)
            used_secondary.add(s_idx)
            
    # Sort final matches by primary index to preserve chronological order
    matches.sort(key=lambda x: x[0])
    return matches
