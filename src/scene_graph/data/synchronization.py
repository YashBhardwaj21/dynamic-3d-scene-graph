"""Timestamp synchronization for TUM RGB-D sequences.

Implements a monotonic nearest-neighbour matching algorithm for timestamps across 
multiple sensors (RGB, Depth, Ground Truth) with a strict maximum time difference (max_dt).
Runs in O(N+M) time and preserves chronological order.
"""

from typing import List, Tuple


def associate(
    primary_timestamps: List[float], 
    secondary_timestamps: List[float], 
    max_dt: float = 0.02
) -> List[Tuple[int, int]]:
    """Monotonic nearest-neighbour timestamp matching.

    Matches timestamps from a primary list to a secondary list.
    Ensures that each primary and secondary timestamp is used at most once.
    Assumes timestamps are strictly monotonically increasing.

    Args:
        primary_timestamps: List of timestamps (e.g., RGB).
        secondary_timestamps: List of timestamps (e.g., Depth).
        max_dt: Maximum allowed time difference in seconds.

    Returns:
        List of matched index pairs (primary_idx, secondary_idx) sorted by primary_idx.
    """
    matches = []
    
    p_idx = 0
    s_idx = 0
    
    while p_idx < len(primary_timestamps) and s_idx < len(secondary_timestamps):
        p_t = primary_timestamps[p_idx]
        
        # Advance secondary index to the closest timestamp
        while s_idx + 1 < len(secondary_timestamps):
            s_t = secondary_timestamps[s_idx]
            s_next = secondary_timestamps[s_idx + 1]
            if abs(p_t - s_next) < abs(p_t - s_t):
                s_idx += 1
            else:
                break
                
        # Now s_idx points to the best match for p_t
        s_t = secondary_timestamps[s_idx]
        diff = abs(p_t - s_t)
        
        if diff <= max_dt:
            # Check if this s_idx was already used by a previous p_idx.
            # If so, we resolve the conflict by keeping the closest match.
            if matches and matches[-1][1] == s_idx:
                prev_p_idx = matches[-1][0]
                prev_diff = abs(primary_timestamps[prev_p_idx] - s_t)
                if diff < prev_diff:
                    # Current match is better, overwrite
                    matches[-1] = (p_idx, s_idx)
            else:
                matches.append((p_idx, s_idx))
                
        p_idx += 1
        
    return matches
