"""Unit tests for Stage 4 & 8 Planner Occupancy and Collision Prevention."""

import numpy as np
import pytest

from scene_graph.navigation.planner import GridPathPlanner


def test_planner_blocks_unknown_cells():
    """Planner must block unknown cells by default (allow_unknown=False)."""
    # 5x5 grid: column 2 is blocked (e.g. unknown unobserved cells)
    costmap = np.zeros((5, 5), dtype=np.uint8)
    costmap[:, 2] = 255  # wall blocking crossing from col 0 to col 4

    planner = GridPathPlanner(costmap, resolution_m=0.1, origin_xy_m=(0.0, 0.0))
    # (2, 2) is not free
    assert not bool(planner.is_free(2, 2))

    res = planner.plan((0.05, 0.25, 0.0), (0.45, 0.25, 0.0))
    assert res.success is False
    assert "No collision-free path exists" in res.rejection_reason


def test_planner_prevents_diagonal_corner_cutting():
    """Planner must not allow diagonal moves that cut between two blocked orthogonal neighbours."""
    # 3x3 grid:
    # [ 0,  X,  0 ]   Start at (0, 0), Goal at (1, 1)
    # [ X,  0,  0 ]   Both (1, 0) and (0, 1) are obstacles!
    # [ 0,  0,  0 ]
    costmap = np.zeros((3, 3), dtype=np.uint8)
    costmap[0, 1] = 255  # (1, 0) in grid (x=1, y=0)
    costmap[1, 0] = 255  # (0, 1) in grid (x=0, y=1)

    planner = GridPathPlanner(costmap, resolution_m=1.0, origin_xy_m=(0.0, 0.0))
    assert bool(planner.is_free(0, 0))
    assert bool(planner.is_free(1, 1))
    assert not bool(planner.is_free(1, 0))
    assert not bool(planner.is_free(0, 1))

    # Start at (0.5, 0.5) [cell 0,0], goal at (1.5, 1.5) [cell 1,1]
    res = planner.plan((0.5, 0.5, 0.0), (1.5, 1.5, 0.0))
    # Direct diagonal from (0,0) to (1,1) is cut off by obstacles at (1,0) and (0,1).
    assert res.success is False


def test_planner_no_silent_goal_shifting():
    """Blocked goal must fail planning instead of silently shifting to an unapproved cell."""
    costmap = np.zeros((5, 5), dtype=np.uint8)
    costmap[3, 3] = 255  # Goal cell is blocked

    planner = GridPathPlanner(costmap, resolution_m=0.1, origin_xy_m=(0.0, 0.0))
    # Goal at (0.35, 0.35) [cell 3, 3]
    res = planner.plan((0.05, 0.05, 0.0), (0.35, 0.35, 0.0), allow_goal_snap=False)
    assert res.success is False
    assert "Silent goal shifting is prohibited" in res.rejection_reason
