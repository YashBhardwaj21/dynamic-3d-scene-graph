"""Interactive Offline Vehicle Simulator & Plotly Visualizer for Dashboard.

Simulates pure-pursuit path following (Mode 1) or direct /cmd_vel kinematics (Mode 2)
on the verified ROS REP-103 occupancy costmap, checking for footprint collisions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import plotly.graph_objects as go
import yaml

from scene_graph.navigation.goal_resolver import GoalResolver
from scene_graph.navigation.planner import GridPathPlanner, PlannedPath
from scene_graph.navigation.controller import DifferentialDriveController, ControllerConfig


@dataclass
class SimulationState:
    """State snapshot at simulation step."""
    time_s: float
    x: float
    y: float
    yaw_rad: float
    linear_v: float
    angular_w: float
    in_collision: bool
    distance_to_goal: float = 0.0


@dataclass
class SimulationResult:
    """Complete trajectory and collision report from simulation run."""
    success: bool
    mode: str
    states: List[SimulationState] = field(default_factory=list)
    has_collision: bool = False
    collision_time_s: Optional[float] = None
    goal_reached: bool = False
    message: str = ""
    target_entity: Optional[str] = None
    target_center: Optional[Tuple[float, float, float]] = None
    planned_path: Optional[PlannedPath] = None


class VehicleSimulator:
    """Offline kinematic simulator for 4-motor differential RC car."""

    def __init__(self, map_dir: Path, robot_radius_m: float = 0.18):
        self.map_dir = Path(map_dir)
        self.robot_radius = robot_radius_m
        self._load_map()

    def _load_map(self):
        """Load occupancy grid, costmap, and semantic entities."""
        # Costmap
        costmap_p = self.map_dir / "navigation_costmap.pgm"
        meta_p = self.map_dir / "navigation_costmap.yaml"
        if not costmap_p.exists():
            costmap_p = self.map_dir / "occupancy_costmap.pgm"
            meta_p = self.map_dir / "occupancy_costmap.yaml"
        if not costmap_p.exists():
            costmap_p = self.map_dir / "map.pgm"
            meta_p = self.map_dir / "map.yaml"

        if not costmap_p.exists():
            raise FileNotFoundError(f"No costmap image found in {self.map_dir}")

        raw_img = cv2.imread(str(costmap_p), cv2.IMREAD_GRAYSCALE)
        # Flip vertically so row 0 corresponds to y_min
        self.grid = np.flipud(raw_img)
        self.height, self.width = self.grid.shape

        with open(meta_p, "r", encoding="utf-8") as f:
            ydata = yaml.safe_load(f)
        self.resolution = float(ydata.get("resolution", 0.05))
        origin = ydata.get("origin", [0.0, 0.0, 0.0])
        self.ox = float(origin[0])
        self.oy = float(origin[1])

        # Semantic entities
        self.entities: Dict[str, Any] = {}
        sem_p = self.map_dir / "semantic_map.json"
        if sem_p.exists():
            with open(sem_p, "r", encoding="utf-8") as f:
                sdata = json.load(f)
                self.entities = sdata.get("entities", {})

    def get_confirmed_entity_names(self) -> List[str]:
        """List verified entities available for targeting."""
        confirmed = []
        for eid, ent in self.entities.items():
            if ent.get("is_confirmed") is True or ent.get("status") in ("CONFIRMED", "ACTIVE"):
                cls = ent.get("class_name", "object")
                confirmed.append(f"{eid} ({cls})")
        return sorted(confirmed)

    def is_collision(self, x: float, y: float) -> bool:
        """Check if robot footprint at (x, y) intersects lethal obstacle on the costmap."""
        gx = int(round((x - self.ox) / self.resolution))
        gy = int(round((y - self.oy) / self.resolution))
        if 0 <= gx < self.width and 0 <= gy < self.height:
            # On ROS costmap (0=lethal obstacle, 254=free, 205=unknown)
            return self.grid[gy, gx] < 128
        return True  # Out of map bounds

    def simulate_navigation_goal(
        self,
        goal_instruction: str,
        start_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        dt: float = 0.05,
        max_duration_s: float = 30.0,
    ) -> SimulationResult:
        """Mode 1: Resolve natural-language prompt or entity ID, plan A* route, simulate pure pursuit."""
        # 1. Resolve goal
        resolver = GoalResolver.from_map_dir(self.map_dir, robot_radius_m=self.robot_radius)
        res_goal = resolver.resolve(goal_instruction, current_robot_pose=start_pose)
        if not res_goal.success:
            return SimulationResult(
                success=False,
                mode="navigation_goal",
                message=f"Goal Resolution Rejected: {res_goal.rejection_reason}",
            )

        # 2. Plan A* path
        planner = GridPathPlanner.from_map_dir(self.map_dir, allow_unknown=True)
        planned = planner.plan(start_pose, res_goal.approach_pose_2d, target_entity_id=res_goal.target_entity_id, allow_goal_snap=True)
        if not planned.success:
            return SimulationResult(
                success=False,
                mode="navigation_goal",
                message=f"Path Planning Failed: {planned.rejection_reason}",
                target_entity=res_goal.target_entity_id,
                target_center=res_goal.target_center_floor,
            )

        # 3. Step-by-step controller simulation
        controller = DifferentialDriveController(
            config=ControllerConfig(
                max_linear_speed_mps=0.30,
                max_angular_speed_radps=1.2,
                lookahead_distance_m=0.35,
                goal_tolerance_pos_m=0.10,
                goal_tolerance_yaw_rad=0.15,
            )
        )

        states: List[SimulationState] = []
        cur_x, cur_y, cur_yaw = start_pose
        t = 0.0
        collided = False
        collision_t = None
        reached = False

        while t < max_duration_s:
            d_goal = math.hypot(res_goal.approach_pose_2d[0] - cur_x, res_goal.approach_pose_2d[1] - cur_y)
            in_col = self.is_collision(cur_x, cur_y)
            if in_col and not collided:
                collided = True
                collision_t = t

            cmd = controller.compute_command((cur_x, cur_y, cur_yaw), planned.waypoints_floor)
            states.append(SimulationState(
                time_s=round(t, 2),
                x=cur_x,
                y=cur_y,
                yaw_rad=cur_yaw,
                linear_v=cmd.linear_velocity_mps,
                angular_w=cmd.angular_velocity_radps,
                in_collision=in_col,
                distance_to_goal=d_goal,
            ))

            if cmd.is_stopped or d_goal <= 0.12:
                reached = True
                break

            # Kinematic integration (unicycle model)
            cur_x += cmd.linear_velocity_mps * math.cos(cur_yaw) * dt
            cur_y += cmd.linear_velocity_mps * math.sin(cur_yaw) * dt
            cur_yaw += cmd.angular_velocity_radps * dt
            # Normalize angle to [-pi, pi]
            cur_yaw = math.atan2(math.sin(cur_yaw), math.cos(cur_yaw))
            t += dt

        msg = "Goal reached successfully!" if reached else ("Collision detected during execution!" if collided else "Simulation completed max duration.")
        return SimulationResult(
            success=reached and not collided,
            mode="navigation_goal",
            states=states,
            has_collision=collided,
            collision_time_s=collision_t,
            goal_reached=reached,
            message=msg,
            target_entity=res_goal.target_entity_id,
            target_center=res_goal.target_center_floor,
            planned_path=planned,
        )

    def simulate_velocity_commands(
        self,
        command_sequence: List[Dict[str, float]],
        start_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        dt: float = 0.05,
    ) -> SimulationResult:
        """Mode 2: Simulate discrete sequence of /cmd_vel {linear_x, angular_z, duration_s}."""
        states: List[SimulationState] = []
        cur_x, cur_y, cur_yaw = start_pose
        t_global = 0.0
        collided = False
        collision_t = None

        for step in command_sequence:
            lin_v = float(step.get("linear_x", 0.0))
            ang_w = float(step.get("angular_z", 0.0))
            dur = float(step.get("duration_s", 1.0))

            t_step = 0.0
            while t_step < dur:
                in_col = self.is_collision(cur_x, cur_y)
                if in_col and not collided:
                    collided = True
                    collision_t = t_global

                states.append(SimulationState(
                    time_s=round(t_global, 2),
                    x=cur_x,
                    y=cur_y,
                    yaw_rad=cur_yaw,
                    linear_v=lin_v,
                    angular_w=ang_w,
                    in_collision=in_col,
                ))

                cur_x += lin_v * math.cos(cur_yaw) * dt
                cur_y += lin_v * math.sin(cur_yaw) * dt
                cur_yaw += ang_w * dt
                cur_yaw = math.atan2(math.sin(cur_yaw), math.cos(cur_yaw))

                t_step += dt
                t_global += dt

        return SimulationResult(
            success=not collided,
            mode="direct_velocity",
            states=states,
            has_collision=collided,
            collision_time_s=collision_t,
            goal_reached=True,
            message="Velocity sequence completed safely." if not collided else f"Collision encountered at t={collision_t:.1f}s!",
        )

    def render_topdown_plotly(
        self,
        sim_result: Optional[SimulationResult] = None,
        current_step_idx: Optional[int] = None,
    ) -> go.Figure:
        """Create rich interactive 2D top-down Plotly figure."""
        fig = go.Figure()

        # 1. Occupancy costmap background
        # Calculate real-world bounds
        x_min = self.ox
        x_max = self.ox + self.width * self.resolution
        y_min = self.oy
        y_max = self.oy + self.height * self.resolution

        # Display costmap as grayscale heatmap
        # 0 = Obstacle (dark), 205 = Unknown (gray), 254 = Free (white)
        fig.add_trace(go.Heatmap(
            z=self.grid,
            x0=x_min,
            dx=self.resolution,
            y0=y_min,
            dy=self.resolution,
            colorscale=[[0, "#1a1a1a"], [0.8, "#808080"], [1.0, "#f0f0f0"]],
            showscale=False,
            hoverinfo="skip",
            name="Occupancy Grid",
        ))

        # 2. Semantic object bounding footprints & markers
        for eid, ent in self.entities.items():
            if ent.get("is_confirmed") is True or ent.get("status") in ("CONFIRMED", "ACTIVE"):
                c = ent.get("centroid_floor", [0, 0, 0])
                cls = ent.get("class_name", "object")
                # Object centroid marker
                fig.add_trace(go.Scatter(
                    x=[c[0]],
                    y=[c[1]],
                    mode="markers+text",
                    marker=dict(size=12, color="#00bcd4", symbol="square"),
                    text=[eid],
                    textposition="top center",
                    textfont=dict(size=10, color="#ffffff"),
                    hovertext=f"Entity: {eid}<br>Class: {cls}<br>Centroid: ({c[0]:.2f}, {c[1]:.2f})",
                    name="Semantic Objects",
                    showlegend=False,
                ))

        # 3. Planned path line (if present)
        if sim_result and sim_result.planned_path and sim_result.planned_path.waypoints_floor:
            pts = sim_result.planned_path.waypoints_floor
            px = [p[0] for p in pts]
            py = [p[1] for p in pts]
            fig.add_trace(go.Scatter(
                x=px,
                y=py,
                mode="lines+markers",
                line=dict(color="#ff9800", width=3, dash="dot"),
                marker=dict(size=6, color="#ff9800"),
                name="Planned A* Path",
            ))

        # 4. Target Standoff Goal Marker
        if sim_result and sim_result.planned_path:
            gx, gy, _ = sim_result.planned_path.goal_pose
            fig.add_trace(go.Scatter(
                x=[gx],
                y=[gy],
                mode="markers",
                marker=dict(size=14, color="#4caf50", symbol="star"),
                name="Approach Goal",
                hovertext=f"Approach Pose: ({gx:.2f}, {gy:.2f})",
            ))

        # 5. Simulated Trajectory Path & Animated Vehicle Pose
        if sim_result and sim_result.states:
            idx = current_step_idx if (current_step_idx is not None and 0 <= current_step_idx < len(sim_result.states)) else (len(sim_result.states) - 1)
            traveled = sim_result.states[: idx + 1]
            tx = [s.x for s in traveled]
            ty = [s.y for s in traveled]

            # Trajectory line
            traj_color = "#e91e63" if sim_result.has_collision else "#00e676"
            fig.add_trace(go.Scatter(
                x=tx,
                y=ty,
                mode="lines",
                line=dict(color=traj_color, width=4),
                name="Travelled Path",
            ))

            # Current vehicle position
            cur = sim_result.states[idx]
            veh_color = "#d50000" if cur.in_collision else "#2979ff"

            # Vehicle circular footprint circle
            theta = np.linspace(0, 2 * np.pi, 30)
            cx = cur.x + self.robot_radius * np.cos(theta)
            cy = cur.y + self.robot_radius * np.sin(theta)
            fig.add_trace(go.Scatter(
                x=cx,
                y=cy,
                mode="lines",
                line=dict(color=veh_color, width=2),
                fill="toself",
                fillcolor="rgba(41, 121, 255, 0.25)" if not cur.in_collision else "rgba(213, 0, 0, 0.4)",
                name="Robot Footprint",
                showlegend=False,
            ))

            # Heading arrow
            hx = cur.x + (self.robot_radius + 0.15) * math.cos(cur.yaw_rad)
            hy = cur.y + (self.robot_radius + 0.15) * math.sin(cur.yaw_rad)
            fig.add_trace(go.Scatter(
                x=[cur.x, hx],
                y=[cur.y, hy],
                mode="lines+markers",
                line=dict(color="#ffd600", width=4),
                marker=dict(size=[8, 12], symbol=["circle", "triangle-up"], color="#ffd600"),
                name="Heading Vector",
                hovertext=f"Pose: ({cur.x:.2f}, {cur.y:.2f})<br>Yaw: {math.degrees(cur.yaw_rad):.1f}°<br>Speed: {cur.linear_v:.2f} m/s",
            ))

        # Format plot
        fig.update_layout(
            title=dict(text="Top-Down Metric Map & Simulation (ROS REP-103 floor_map)", font=dict(size=15)),
            xaxis=dict(title="X (meters) [Forward]", range=[x_min, x_max], constrain="domain"),
            yaxis=dict(title="Y (meters) [Left]", range=[y_min, y_max], scaleanchor="x", scaleratio=1),
            width=850,
            height=650,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig
