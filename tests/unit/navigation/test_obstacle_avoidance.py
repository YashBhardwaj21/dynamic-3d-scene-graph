"""Automated unit and integration tests for Obstacle Detection, Safety, and Dynamic Replanning.

Tests Cases:
  Case E: Obstacle on the route: car stops, updates costmap, replans around obstacle.
  Case F: No safe alternate route: car remains stopped, reports blocked, retains original target.
  Case I: Sensor failure: depth frames stale or unavailable -> safety system stops vehicle.
  Corridor safety checks: Footprint width, noise rejection, distance thresholding.
"""

import math
import time
import pytest
import numpy as np

from scene_graph.navigation.obstacle_monitor import (
    LiveObstacleMonitor,
    ObstacleMonitorConfig,
    ObstacleSafetyStatus,
)
from scene_graph.navigation.planner import GridPathPlanner


@pytest.fixture
def empty_grid_planner():
    # 10m x 10m costmap at 0.05m resolution (200x200 grid)
    res = 0.05
    width = 200
    height = 200
    grid = np.zeros((height, width), dtype=np.uint8)  # 0 = free
    return GridPathPlanner(grid, res, (-5.0, -5.0), occupied_threshold=100)


def test_obstacle_monitor_clear_path():
    """No obstacle in forward corridor."""
    cfg = ObstacleMonitorConfig(critical_stop_distance_m=0.40)
    monitor = LiveObstacleMonitor(cfg)

    # Clean depth frame: depth is 2.5m everywhere (far away)
    depth = np.full((480, 640), 2500, dtype=np.uint16)
    monitor.update_depth_frame(depth, timestamp_s=time.time())

    rep = monitor.evaluate_safety(current_pose_floor=(0.0, 0.0, 0.0))
    assert rep.status == ObstacleSafetyStatus.CLEAR
    assert rep.is_safe_to_proceed is True


def test_obstacle_monitor_blocked_corridor():
    """Obstacle directly inside forward motion corridor triggers BLOCKED status."""
    cfg = ObstacleMonitorConfig(critical_stop_distance_m=0.40)
    monitor = LiveObstacleMonitor(cfg)

    depth = np.full((480, 640), 2500, dtype=np.uint16)
    # Put cluster of obstacle points at 0.32m in center of view (corridor)
    depth[200:280, 280:360] = 320  # 320 mm = 0.32 m
    monitor.update_depth_frame(depth, timestamp_s=time.time())

    rep = monitor.evaluate_safety(current_pose_floor=(0.0, 0.0, 0.0))
    assert rep.status == ObstacleSafetyStatus.BLOCKED
    assert rep.is_safe_to_proceed is False
    assert rep.closest_distance_m <= 0.40
    assert rep.obstacle_point_floor is not None


def test_obstacle_monitor_lateral_obstacle_ignored():
    """Obstacle outside lateral corridor (to the far left/right) does not block forward corridor."""
    cfg = ObstacleMonitorConfig(corridor_width_m=0.40, critical_stop_distance_m=0.40)
    monitor = LiveObstacleMonitor(cfg)

    depth = np.full((480, 640), 2500, dtype=np.uint16)
    # Put obstacle points at far edge of image (X_cam > 0.5m lateral)
    depth[200:280, 10:50] = 320
    monitor.update_depth_frame(depth, timestamp_s=time.time())

    rep = monitor.evaluate_safety(current_pose_floor=(0.0, 0.0, 0.0))
    # Outside corridor width -> path is clear for forward motion
    assert rep.status == ObstacleSafetyStatus.CLEAR
    assert rep.is_safe_to_proceed is True


def test_case_i_sensor_failure_stale_depth():
    """Case I: Stale or missing depth frames trigger SENSOR_FAILURE and stop vehicle."""
    cfg = ObstacleMonitorConfig(max_stale_duration_s=0.20)
    monitor = LiveObstacleMonitor(cfg)

    # 1. No frames received at all
    rep1 = monitor.evaluate_safety(current_pose_floor=(0.0, 0.0, 0.0))
    assert rep1.status == ObstacleSafetyStatus.SENSOR_FAILURE
    assert rep1.is_safe_to_proceed is False

    # 2. Frame received 1 second ago (stale > 0.20s)
    depth = np.full((480, 640), 2000, dtype=np.uint16)
    monitor.update_depth_frame(depth, timestamp_s=time.time() - 1.0)
    rep2 = monitor.evaluate_safety(current_pose_floor=(0.0, 0.0, 0.0))
    assert rep2.status == ObstacleSafetyStatus.SENSOR_FAILURE
    assert rep2.is_safe_to_proceed is False


def test_case_e_obstacle_replanning_with_detour(empty_grid_planner):
    """Case E: Dynamic obstacle detected on route; system injects obstacle and finds detour."""
    planner = empty_grid_planner
    start_pose = (0.0, 0.0, 0.0)
    goal_pose = (2.0, 0.0, 0.0)

    # 1. Initial direct path
    initial_path = planner.plan(start_pose, goal_pose)
    assert initial_path.success is True
    assert initial_path.path_length_m == pytest.approx(2.0, abs=0.1)

    # 2. Obstacle suddenly detected right in the middle at (1.0, 0.0)
    planner.inject_obstacle(world_x=1.0, world_y=0.0, radius_m=0.25, inflation_radius_m=0.15)

    # Verify that cell is now impassable
    gx, gy = planner.world_to_grid(1.0, 0.0)
    assert not planner.is_free(gx, gy)

    # 3. Replan from current pose to same original goal
    replan_path = planner.plan(start_pose, goal_pose)
    assert replan_path.success is True, "Planner must find alternate route around the dynamic obstacle!"
    # The path must curve around the obstacle, so it is longer than the direct 2.0m line
    assert replan_path.path_length_m > 2.05

    # Check that none of the new path waypoints intersect the obstacle cell
    for pt in replan_path.waypoints_floor:
        wx, wy = pt[0], pt[1]
        dist_to_obs = math.hypot(wx - 1.0, wy - 0.0)
        assert dist_to_obs > 0.20, "Detour must safely clear the injected obstacle!"


def test_case_f_blocked_without_alternate_route(empty_grid_planner):
    """Case F: Obstacle completely seals off a narrow corridor; planner reports blocked and stays stopped."""
    planner = empty_grid_planner
    start_pose = (0.0, 0.0, 0.0)
    goal_pose = (3.0, 0.0, 0.0)

    # Build a wall across X=1.5 from Y=-5 to Y=+5 leaving no passage
    for y in np.arange(-5.0, 5.0, 0.1):
        planner.inject_obstacle(world_x=1.5, world_y=float(y), radius_m=0.15, inflation_radius_m=0.05)

    # Replan attempt
    replan = planner.plan(start_pose, goal_pose)
    assert replan.success is False
    assert "No path found" in replan.rejection_reason or "collision" in replan.rejection_reason
