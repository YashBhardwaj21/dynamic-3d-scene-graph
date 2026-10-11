"""Unit tests for A* Path Planner and Differential-Drive Skid-Steer Controller."""

import math
import numpy as np
import pytest

from scene_graph.navigation.planner import GridPathPlanner, PlannedPath
from scene_graph.navigation.controller import DifferentialDriveController, ControllerConfig, MotionCommand


def test_grid_planner_simple_corridor():
    """Planner must find collision-free path through open corridor between obstacles."""
    # 50x50 grid at 0.1m/cell = 5.0m x 5.0m
    grid = np.zeros((50, 50), dtype=np.uint8)
    # Add an obstacle wall in middle with a gap
    grid[20:30, 0:35] = 255  # Wall leaving gap on right (x: 35..49)

    planner = GridPathPlanner(grid, resolution_m=0.1, origin_xy_m=(0.0, 0.0))

    start = (1.0, 1.0, 0.0)
    goal = (1.0, 4.0, math.pi / 2)

    path = planner.plan(start, goal)
    assert path.success is True
    assert len(path.waypoints_floor) >= 2
    assert path.path_length_m > 3.0  # Must navigate around obstacle

    # Verify no waypoint is inside obstacle
    for wx, wy, _ in path.waypoints_floor:
        gx, gy = planner.world_to_grid(wx, wy)
        assert grid[gy, gx] == 0, f"Waypoint ({wx}, {wy}) is inside obstacle!"


def test_grid_planner_blocked_goal():
    """Planner must report failure if goal is completely blocked by obstacles."""
    grid = np.zeros((30, 30), dtype=np.uint8)
    grid[10:20, 10:20] = 255  # Box obstacle

    planner = GridPathPlanner(grid, resolution_m=0.1, origin_xy_m=(0.0, 0.0))
    start = (0.5, 0.5, 0.0)
    goal = (1.5, 1.5, 0.0)  # Inside middle of box

    # Nearest free search might find cell if within max_radius, but if surrounded by wall:
    walled_grid = np.zeros((40, 40), dtype=np.uint8)
    walled_grid[5:35, 5:35] = 255  # Large obstacle covering goal
    planner2 = GridPathPlanner(walled_grid, resolution_m=0.1, origin_xy_m=(0.0, 0.0))
    path = planner2.plan(start, (2.0, 2.0, 0.0))
    assert path.success is False
    assert "blocked by obstacles" in path.rejection_reason or "impassable" in path.rejection_reason


def test_differential_drive_controller_tracking():
    """Controller must output positive linear velocity and steer toward target waypoint."""
    cfg = ControllerConfig(max_linear_speed_mps=0.30, track_width_m=0.20)
    controller = DifferentialDriveController(cfg)

    waypoints = [(0.0, 0.0, 0.0), (1.0, 0.5, 0.0), (2.0, 0.5, 0.0)]
    cur_pose = (0.0, 0.0, 0.0)

    cmd = controller.compute_command(cur_pose, waypoints)
    assert cmd.is_stopped is False
    assert cmd.linear_velocity_mps > 0.05
    assert cmd.angular_velocity_radps > 0.0  # Must turn left toward (1.0, 0.5)
    assert cmd.v_right_mps > cmd.v_left_mps  # Right wheel faster for left turn


def test_controller_arrival_stop():
    """Controller must stop when within position and yaw tolerance."""
    cfg = ControllerConfig(goal_tolerance_pos_m=0.08, goal_tolerance_yaw_rad=0.15)
    controller = DifferentialDriveController(cfg)

    waypoints = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.5)]
    # Robot is already at the goal within tolerances
    cur_pose = (1.02, 0.99, 0.52)

    cmd = controller.compute_command(cur_pose, waypoints)
    assert cmd.is_stopped is True
    assert cmd.phase == "STOPPED"
    assert cmd.linear_velocity_mps == 0.0
    assert cmd.angular_velocity_radps == 0.0
