"""A* Grid Path Planner and Route Visualizer for Differential-Drive Robot Navigation.

Finds collision-free paths from current robot pose to goal pose in the floor-aligned costmap.
Generates publication-quality route visualizations and checks safety clearances.
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


@dataclass
class PlannedPath:
    """Complete planned path and trajectory metrics."""
    success: bool
    waypoints_floor: List[Tuple[float, float, float]] = field(default_factory=list)  # [(x, y, yaw)]
    path_length_m: float = 0.0
    turn_points_count: int = 0
    start_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_entity_id: Optional[str] = None
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "path_length_m": round(self.path_length_m, 3),
            "turn_points_count": self.turn_points_count,
            "waypoints_count": len(self.waypoints_floor),
            "start_pose": {"x": self.start_pose[0], "y": self.start_pose[1], "yaw_rad": self.start_pose[2]},
            "goal_pose": {"x": self.goal_pose[0], "y": self.goal_pose[1], "yaw_rad": self.goal_pose[2]},
            "target_entity_id": self.target_entity_id,
            "waypoints_floor": [
                {"x": round(w[0], 3), "y": round(w[1], 3), "yaw_rad": round(w[2], 3)}
                for w in self.waypoints_floor
            ],
            "rejection_reason": self.rejection_reason,
        }


class GridPathPlanner:
    """A* Path Planner on 2D costmaps following ROS REP-103 convention."""

    def __init__(
        self,
        costmap_grid: np.ndarray,
        resolution_m: float,
        origin_xy_m: Tuple[float, float],
        occupied_threshold: int = 100,
    ):
        """
        costmap_grid: 2D numpy array where:
          0 = free space
          255 (or > occupied_threshold) = lethal obstacle / inflated collision zone
        origin_xy_m: (min_x_m, min_y_m) world coordinate of grid[0, 0]
        """
        self.grid = costmap_grid
        self.height, self.width = costmap_grid.shape
        self.res = resolution_m
        self.ox, self.oy = origin_xy_m
        self.occ_thresh = occupied_threshold

    @classmethod
    def from_map_dir(cls, map_dir: Path) -> "GridPathPlanner":
        map_dir = Path(map_dir)
        # Prefer inflated navigation costmap if present
        costmap_p = map_dir / "occupancy_costmap.pgm"
        meta_p = map_dir / "occupancy_costmap.yaml"
        if not costmap_p.exists():
            costmap_p = map_dir / "navigation_costmap.pgm"
            meta_p = map_dir / "navigation_costmap.yaml"
        if not costmap_p.exists():
            costmap_p = map_dir / "occupancy_map.pgm"
            meta_p = map_dir / "occupancy_map.yaml"
        if not costmap_p.exists():
            costmap_p = map_dir / "map.pgm"
            meta_p = map_dir / "map.yaml"

        if not costmap_p.exists():
            raise FileNotFoundError(f"No costmap or occupancy map found in {map_dir}")

        img = cv2.imread(str(costmap_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read costmap image from {costmap_p}")

        # Parse YAML
        import yaml
        with open(meta_p, "r", encoding="utf-8") as f:
            ydata = yaml.safe_load(f)
        res = float(ydata.get("resolution", 0.05))
        origin = ydata.get("origin", [0.0, 0.0, 0.0])
        ox, oy = float(origin[0]), float(origin[1])

        # In ROS PGM, images are stored with origin at bottom-left, so rows are flipped.
        # Flip vertically so row 0 corresponds to y_min (origin_y)
        img_aligned = np.flipud(img)

        # In ROS PGM: 0=occupied (black), 205=unknown (grey), 254=free (white)
        # Black (< 128) are lethal/inflated obstacles (cost 255)
        obstacle_mask = np.zeros_like(img_aligned, dtype=np.uint8)
        obstacle_mask[img_aligned < 128] = 255

        return cls(obstacle_mask, res, (ox, oy))

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(round((x - self.ox) / self.res))
        gy = int(round((y - self.oy) / self.res))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        x = self.ox + (gx + 0.5) * self.res
        y = self.oy + (gy + 0.5) * self.res
        return x, y

    def is_free(self, gx: int, gy: int) -> bool:
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return self.grid[gy, gx] < self.occ_thresh
        return False

    def find_nearest_free_cell(self, gx: int, gy: int, max_radius: int = 5) -> Optional[Tuple[int, int]]:
        """If goal is slightly inside an inflated cell border, find closest reachable free cell."""
        if self.is_free(gx, gy):
            return gx, gy
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        nx, ny = gx + dx, gy + dy
                        if self.is_free(nx, ny):
                            return nx, ny
        return None

    def plan(
        self,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
        target_entity_id: Optional[str] = None,
    ) -> PlannedPath:
        """Execute A* search on costmap from start_pose (x, y, yaw) to goal_pose (x, y, yaw)."""
        sx, sy, syaw = start_pose
        gx, gy, gyaw = goal_pose

        sgx, sgy = self.world_to_grid(sx, sy)
        ggx, ggy = self.world_to_grid(gx, gy)

        # Check and resolve start cell
        if not self.is_free(sgx, sgy):
            free_start = self.find_nearest_free_cell(sgx, sgy)
            if not free_start:
                return PlannedPath(
                    success=False,
                    start_pose=start_pose,
                    goal_pose=goal_pose,
                    rejection_reason=f"Start position ({sx:.2f}, {sy:.2f}) is in an impassable collision zone",
                )
            sgx, sgy = free_start

        # Check and resolve goal cell
        if not self.is_free(ggx, ggy):
            free_goal = self.find_nearest_free_cell(ggx, ggy)
            if not free_goal:
                return PlannedPath(
                    success=False,
                    start_pose=start_pose,
                    goal_pose=goal_pose,
                    rejection_reason=f"Goal position ({gx:.2f}, {gy:.2f}) is blocked by obstacles or outside navigable area",
                )
            ggx, ggy = free_goal

        # 8-connected A* search
        # Actions: (dx, dy, cost)
        actions = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, -1, math.sqrt(2))
        ]

        open_set = []
        heapq.heappush(open_set, (0.0, 0.0, (sgx, sgy)))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_cost: Dict[Tuple[int, int], float] = {(sgx, sgy): 0.0}

        def heuristic(cell_a, cell_b):
            dx = abs(cell_a[0] - cell_b[0])
            dy = abs(cell_a[1] - cell_b[1])
            return (dx + dy) + (math.sqrt(2) - 2.0) * min(dx, dy)

        reached = False
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            if current == (ggx, ggy):
                reached = True
                break

            for adx, ady, step_cost in actions:
                neighbor = (current[0] + adx, current[1] + ady)
                if not self.is_free(neighbor[0], neighbor[1]):
                    continue

                tentative_g = current_g + step_cost
                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    priority = tentative_g + heuristic(neighbor, (ggx, ggy))
                    heapq.heappush(open_set, (priority, tentative_g, neighbor))
                    came_from[neighbor] = current

        if not reached:
            return PlannedPath(
                success=False,
                start_pose=start_pose,
                goal_pose=goal_pose,
                rejection_reason="No collision-free path exists between current robot pose and target goal",
            )

        # Reconstruct path
        path_cells = [(ggx, ggy)]
        curr = (ggx, ggy)
        while curr in came_from:
            curr = came_from[curr]
            path_cells.append(curr)
        path_cells.reverse()

        # Convert to world coordinates and simplify waypoints
        raw_pts = [self.grid_to_world(c[0], c[1]) for c in path_cells]
        smoothed_pts = self._simplify_path(raw_pts)

        # Compute headings for each waypoint
        waypoints = []
        total_len = 0.0
        for i in range(len(smoothed_pts)):
            px, py = smoothed_pts[i]
            if i < len(smoothed_pts) - 1:
                nx, ny = smoothed_pts[i + 1]
                yaw = math.atan2(ny - py, nx - px)
                total_len += math.hypot(nx - px, ny - py)
            else:
                yaw = gyaw  # Align with final goal orientation
            waypoints.append((px, py, yaw))

        turn_count = max(0, len(smoothed_pts) - 2)

        return PlannedPath(
            success=True,
            waypoints_floor=waypoints,
            path_length_m=total_len,
            turn_points_count=turn_count,
            start_pose=start_pose,
            goal_pose=(waypoints[-1][0], waypoints[-1][1], gyaw),
            target_entity_id=target_entity_id,
        )

    def _simplify_path(self, pts: List[Tuple[float, float]], tolerance_m: float = 0.08) -> List[Tuple[float, float]]:
        """Line-of-sight path simplification to eliminate zig-zags."""
        if len(pts) <= 2:
            return pts

        simplified = [pts[0]]
        current_idx = 0

        while current_idx < len(pts) - 1:
            furthest_idx = current_idx + 1
            for test_idx in range(current_idx + 2, len(pts)):
                if self._line_of_sight(pts[current_idx], pts[test_idx]):
                    furthest_idx = test_idx
                else:
                    break
            simplified.append(pts[furthest_idx])
            current_idx = furthest_idx

        return simplified

    def _line_of_sight(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """Bresenham line check on costmap grid."""
        gx1, gy1 = self.world_to_grid(p1[0], p1[1])
        gx2, gy2 = self.world_to_grid(p2[0], p2[1])

        dx = abs(gx2 - gx1)
        dy = abs(gy2 - gy1)
        sx = 1 if gx1 < gx2 else -1
        sy = 1 if gy1 < gy2 else -1
        err = dx - dy

        x, y = gx1, gy1
        while True:
            if not self.is_free(x, y):
                return False
            if x == gx2 and y == gy2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return True

    def visualize_planned_route(
        self,
        planned_path: PlannedPath,
        output_image_path: Path,
        target_center: Optional[Tuple[float, float, float]] = None,
        target_radius_m: float = 0.4,
        robot_radius_m: float = 0.18,
    ) -> None:
        """Render publication-grade visualization of costmap, robot footprint, and planned route."""
        fig, ax = plt.subplots(figsize=(10, 10), dpi=200)

        # Plot costmap background (white=free, dark=obstacle)
        extent = [self.ox, self.ox + self.width * self.res, self.oy, self.oy + self.height * self.res]
        display_grid = np.where(self.grid >= self.occ_thresh, 0.0, 1.0)
        ax.imshow(display_grid, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1, alpha=0.85)

        # Plot target object footprint and center
        if target_center is not None:
            tcx, tcy = target_center[0], target_center[1]
            obj_circle = patches.Circle(
                (tcx, tcy), target_radius_m,
                color="tab:purple", alpha=0.45, label=f"Target Object ({planned_path.target_entity_id or 'Goal'})"
            )
            ax.add_patch(obj_circle)
            ax.plot(tcx, tcy, "x", color="purple", markersize=10, mew=2)

        if planned_path.success and planned_path.waypoints_floor:
            xs = [w[0] for w in planned_path.waypoints_floor]
            ys = [w[1] for w in planned_path.waypoints_floor]

            # Route line
            ax.plot(xs, ys, "-", color="tab:cyan", linewidth=3.0, label=f"Planned Route ({planned_path.path_length_m:.2f} m)")
            ax.plot(xs, ys, "o", color="tab:blue", markersize=4, label="Turn Waypoints")

            # Start robot footprint
            sx, sy, syaw = planned_path.start_pose
            start_circle = patches.Circle((sx, sy), robot_radius_m, color="tab:green", alpha=0.5, label="Start Robot Pose")
            ax.add_patch(start_circle)
            ax.arrow(sx, sy, 0.3 * math.cos(syaw), 0.3 * math.sin(syaw), head_width=0.08, color="darkgreen")

            # Final goal pose and footprint
            gx, gy, gyaw = planned_path.goal_pose
            goal_circle = patches.Circle((gx, gy), robot_radius_m, color="tab:red", alpha=0.5, label="Approach Standoff Goal")
            ax.add_patch(goal_circle)
            ax.arrow(gx, gy, 0.3 * math.cos(gyaw), 0.3 * math.sin(gyaw), head_width=0.08, color="darkred")

        ax.set_title(f"Planned Autonomous Navigation Route — Goal: {planned_path.target_entity_id or 'Goal Pose'}", fontsize=13, weight="bold")
        ax.set_xlabel("Floor Coordinate X (meters - Forward)", fontsize=11)
        ax.set_ylabel("Floor Coordinate Y (meters - Left)", fontsize=11)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.axis("equal")

        output_image_path = Path(output_image_path)
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(str(output_image_path))
        plt.close(fig)
