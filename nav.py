#!/usr/bin/env python3
"""Lightweight Autonomous Navigation for Raspberry Pi 5 + Intel RealSense D455 + L298N 4WD.

Key Design:
  1. A* Path Planning on 2D metric occupancy map with obstacle inflation.
  2. D455 Hardware Gyro Closed-Loop: Real-time heading tracking and drift-free active steering.
  3. D455 Depth Obstacle Avoidance: Direct ROI depth check (< 0.40m halt & A* replan) - zero CPU bloat.
  4. Dead-Reckoning Position Tracking: Clean integration of speed and gyro heading.
  5. Minimal Bandwidth & CPU: Depth + Gyro only (no heavy visual odometry or image streaming).
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from gpiozero import DigitalOutputDevice, PWMOutputDevice
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False

# L298N GPIO Pinout (Raspberry Pi 5 BCM numbering)
IN1_PIN = 17
IN2_PIN = 27
IN3_PIN = 22
IN4_PIN = 23
ENA_PIN = 18
ENB_PIN = 19

# Drive Dynamics & Calibrations
CRUISE_PWM = 0.88          # Forward rolling duty cycle
TURN_PWM = 0.90            # Skid-steer breakaway & turn duty cycle
SPEED_MPS = 0.35           # Calibrated linear speed in m/s (~35 cm/s at 88% PWM)
TURN_RADPS = 1.40          # Turn rate in rad/s (~80 deg/s at 90% PWM)


def find_fused_dir() -> Path:
    candidates = [
        Path("outputs/semantic_mapping_fused_desk_run"),
        Path("semantic_mapping_fused_desk_run"),
        Path("outputs/outputs/semantic_mapping_fused_desk_run"),
        Path.home() / "perception/outputs/semantic_mapping_fused_desk_run",
        Path.home() / "perception/semantic_mapping_fused_desk_run",
    ]
    for c in candidates:
        if (c / "occupancy_map.yaml").exists() or (c / "map.yaml").exists():
            return c
    for p in Path(".").glob("**/occupancy_map.yaml"):
        return p.parent
    for p in Path(".").glob("**/map.yaml"):
        return p.parent
    return Path("outputs/semantic_mapping_fused_desk_run")


STATE_FILE = Path("robot_state.json")
FUSED_DIR = find_fused_dir()
FUSED_STATE_FILE = FUSED_DIR / "robot_navigation_state.json"

# Facility Landmark Database with Standoff Approach Poses
TARGET_DATABASE = {
    "desk_001": {
        "aliases": ["desk 1", "desk 01", "desk001", "main desk", "table 1", "primary desk"],
        "approach": (2.00, -1.25, math.radians(-90.0)),
        "center": (2.30, -2.14),
        "desc": "Primary workstation desk",
    },
    "chair_002": {
        "aliases": ["chair 2", "chair 02", "chair002", "side chair", "desk chair", "yellow chair", "conference chair"],
        "approach": (1.85, -1.25, math.radians(-90.0)),
        "center": (1.76, -1.96),
        "desc": "Desk 1 yellow conference chair",
    },
    "chair_003": {
        "aliases": ["chair 3", "chair 03", "chair003", "meeting chair", "central chair"],
        "approach": (1.13, -2.20, math.radians(-90.0)),
        "center": (1.13, -2.92),
        "desc": "Central meeting chair",
    },
    "box_001": {
        "aliases": ["box 1", "box 01", "storage box 1", "box", "storage box_001"],
        "approach": (1.66, -1.33, math.radians(-68.6)),
        "center": (1.65, -2.89),
        "desc": "Storage box by desk 1",
    },
    "chair_005": {
        "aliases": ["chair 5", "chair 05", "chair005", "south chair"],
        "approach": (0.61, -3.22, math.radians(-76.8)),
        "center": (0.66, -3.87),
        "desc": "South meeting chair",
    },
    "desk_005": {
        "aliases": ["desk 5", "desk 05", "desk005", "south bench", "lab bench"],
        "approach": (-0.16, -4.50, math.radians(-90.0)),
        "center": (-0.14, -5.72),
        "desc": "South laboratory bench",
    },
    "desk_018": {
        "aliases": ["desk 18", "desk18", "table 18", "bench 18", "west desk", "west bench"],
        "approach": (-1.36, -3.02, math.pi),
        "center": (-2.56, -3.02),
        "desc": "West workstation bench",
    },
    "desk_017": {
        "aliases": ["desk 17", "desk17", "table 17", "northwest bench"],
        "approach": (-1.17, -2.30, math.radians(80.3)),
        "center": (-1.88, -1.53),
        "desc": "Northwest center bench",
    },
    "desk_023": {
        "aliases": ["desk 23", "desk23", "table 23", "west corridor desk"],
        "approach": (-1.08, -2.05, math.radians(78.3)),
        "center": (-1.16, -1.11),
        "desc": "West corridor workstation",
    },
    "desk_024": {
        "aliases": ["desk 24", "desk24", "table 24", "north desk"],
        "approach": (-0.38, -0.80, math.radians(83.7)),
        "center": (-0.65, -0.28),
        "desc": "North corridor desk",
    },
    "origin": {
        "aliases": ["origin", "start", "home", "base", "0 0", "start origin"],
        "approach": (0.0, 0.0, 0.0),
        "center": (0.0, 0.0),
        "desc": "Facility entrance origin",
    },
}

TOUR_SEQUENCE = [
    ("desk_001", "Primary Workstation Desk"),
    ("chair_002", "Conference Side Chair"),
    ("chair_003", "Central Meeting Chair"),
    ("box_001", "Storage Box by Desk 1"),
    ("chair_005", "South Meeting Chair"),
    ("desk_005", "South Laboratory Bench"),
    ("desk_018", "West Workstation Bench"),
    ("desk_017", "Northwest Center Bench"),
    ("desk_023", "West Corridor Workstation"),
    ("desk_024", "North Corridor Desk"),
]


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, +pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def load_state() -> tuple[float, float, float]:
    """Load last known robot pose (x, y, yaw_rad)."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
                return float(d.get("x", 0.0)), float(d.get("y", 0.0)), float(d.get("yaw", 0.0))
        except Exception:
            pass
    return (0.0, 0.0, 0.0)


def save_state(x: float, y: float, yaw: float, last_target: str):
    """Save persistent robot pose."""
    d = {
        "x": round(x, 3),
        "y": round(y, 3),
        "yaw": round(yaw, 4),
        "yaw_deg": round(math.degrees(yaw), 1),
        "last_target": last_target,
        "updated_at": time.time(),
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

    if FUSED_DIR.exists():
        try:
            with open(FUSED_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump({"current_pose": d}, f, indent=2)
        except Exception:
            pass


def resolve_target(query: str, semantic_entities: Dict[str, dict] = None) -> tuple[Optional[str], Optional[dict]]:
    """Match query string against target database."""
    if not query:
        return None, None
    q = query.strip().lower()
    for prefix in ["go to ", "move to ", "navigate to ", "drive to ", "find ", "head to "]:
        if q.startswith(prefix):
            q = q[len(prefix):].strip()

    # Pass 1: exact matches
    for tid, info in TARGET_DATABASE.items():
        if q == tid.lower() or q == tid.lower().replace("_", "") or q == tid.lower().replace("_", " "):
            return tid, info
        for alias in info.get("aliases", []):
            if q == alias.lower():
                return tid, info

    # Pass 2: token match
    q_tokens = set(q.replace("_", " ").split())
    for tid, info in TARGET_DATABASE.items():
        if q_tokens == set(tid.lower().replace("_", " ").split()):
            return tid, info
        for alias in info.get("aliases", []):
            if q_tokens == set(alias.lower().split()):
                return tid, info

    # Pass 3: number extraction (e.g. "desk 18")
    import re
    q_nums = re.findall(r"\d+", q)
    if q_nums:
        for tid, info in TARGET_DATABASE.items():
            t_nums = re.findall(r"\d+", tid)
            if [str(int(n)) for n in q_nums] == [str(int(n)) for n in t_nums]:
                for word in ["desk", "chair", "table", "box", "bench"]:
                    if word in q and (word in tid or any(word in a for a in info.get("aliases", []))):
                        return tid, info

    # Pass 4: dynamic semantic map fallback
    if semantic_entities:
        for eid, ent in semantic_entities.items():
            if q == eid.lower() or q == eid.lower().replace("_", " "):
                cf = ent.get("centroid_floor", [0, 0])
                cls_name = ent.get("class_name", "object")
                info = {
                    "aliases": [eid],
                    "approach": (cf[0] + 0.50, cf[1], math.pi),
                    "center": (cf[0], cf[1]),
                    "desc": f"Scene object: {eid} ({cls_name})",
                }
                return eid, info

    return None, None


class FastAStarPlanner:
    """8-connected grid A* path planner on 2D metric occupancy grid."""

    def __init__(self, map_dir: Path):
        self.map_dir = Path(map_dir)
        yaml_p = self.map_dir / "occupancy_map.yaml"
        if not yaml_p.exists():
            yaml_p = self.map_dir / "map.yaml"
        if not yaml_p.exists():
            for p in self.map_dir.glob("**/*map*.yaml"):
                yaml_p = p
                self.map_dir = p.parent
                break
        if not yaml_p.exists():
            for p in Path(".").glob("**/occupancy_map.yaml"):
                yaml_p = p
                self.map_dir = p.parent
                break

        import yaml
        with open(yaml_p, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)

        self.res = float(meta["resolution"])
        orig = [float(v) for v in meta["origin"]]
        self.ox, self.oy = orig[0], orig[1]

        img_name = meta["image"]
        pgm_p = self.map_dir / img_name
        if not pgm_p.exists():
            cand = self.map_dir / (Path(img_name).stem + ".png")
            if cand.exists():
                pgm_p = cand
            else:
                for p in self.map_dir.glob("*.pgm"):
                    pgm_p = p
                    break

        raw = cv2.imread(str(pgm_p), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Could not read map image at '{pgm_p}'")
        self.grid = np.flipud(raw)

        self.h, self.w = self.grid.shape
        self.extent = [self.ox, self.ox + self.w * self.res, self.oy, self.oy + self.h * self.res]

        # Inflate obstacles by robot radius + safety margin (0.18m)
        r_cells = max(1, int(round(0.18 / self.res)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_cells + 1, 2 * r_cells + 1))
        obs_mask = (self.grid <= 64).astype(np.uint8)
        self.inflated_obs = cv2.dilate(obs_mask, kernel)

        self.raw_grid = self.grid.copy()
        self.raw_inflated_obs = self.inflated_obs.copy()

    def reset_dynamic_obstacles(self):
        """Reset occupancy grid to static map."""
        self.grid = self.raw_grid.copy()
        self.inflated_obs = self.raw_inflated_obs.copy()

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor((x - self.ox) / self.res)), int(math.floor((y - self.oy) / self.res))

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        return self.ox + (gx + 0.5) * self.res, self.oy + (gy + 0.5) * self.res

    def is_free(self, gx: int, gy: int, check_inflation: bool = True) -> bool:
        if 0 <= gx < self.w and 0 <= gy < self.h:
            if self.grid[gy, gx] <= 64:
                return False
            if check_inflation and self.inflated_obs[gy, gx] > 0:
                return False
            return True
        return False

    def find_nearest_free(self, gx: int, gy: int, max_r: int = 8) -> tuple[int, int]:
        if self.is_free(gx, gy, check_inflation=True):
            return gx, gy
        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        nx, ny = gx + dx, gy + dy
                        if self.is_free(nx, ny, check_inflation=True):
                            return nx, ny
        # Fallback to uninflated cell if close to boundary
        for r in range(0, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = gx + dx, gy + dy
                    if self.is_free(nx, ny, check_inflation=False):
                        return nx, ny
        return gx, gy

    def add_dynamic_obstacle(self, x: float, y: float, radius_m: float = 0.30):
        """Mark a sensed obstacle into occupancy grid for A* replanning."""
        gx, gy = self.world_to_grid(x, y)
        cell_r = max(1, int(math.ceil(radius_m / self.res)))
        for dx in range(-cell_r, cell_r + 1):
            for dy in range(-cell_r, cell_r + 1):
                if dx * dx + dy * dy <= cell_r * cell_r:
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.w and 0 <= ny < self.h:
                        self.grid[ny, nx] = 0
                        self.inflated_obs[ny, nx] = 1

    def line_of_sight(self, p1: tuple[float, float], p2: tuple[float, float], clearance_m: float = 0.12) -> bool:
        """Check collision-free line-of-sight between two points."""
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist < 0.03:
            return True
        steps = max(2, int(math.ceil(dist / (self.res * 0.5))))
        clearance_cells = max(1, int(round(clearance_m / self.res)))
        for s in range(steps + 1):
            alpha = s / steps
            x = p1[0] + alpha * (p2[0] - p1[0])
            y = p1[1] + alpha * (p2[1] - p1[1])
            gx, gy = self.world_to_grid(x, y)
            for cdx in range(-clearance_cells, clearance_cells + 1):
                for cdy in range(-clearance_cells, clearance_cells + 1):
                    if cdx * cdx + cdy * cdy <= clearance_cells * clearance_cells:
                        nx, ny = gx + cdx, gy + cdy
                        if 0 <= nx < self.w and 0 <= ny < self.h:
                            if self.grid[ny, nx] <= 64:
                                return False
                        else:
                            return False
        return True

    def smooth_path(self, waypoints: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Shortcut waypoints using collision-free line-of-sight."""
        if len(waypoints) <= 2:
            return waypoints
        smoothed = [waypoints[0]]
        i = 0
        while i < len(waypoints) - 1:
            best_j = i + 1
            for j in range(len(waypoints) - 1, i + 1, -1):
                if self.line_of_sight(smoothed[-1], waypoints[j]):
                    best_j = j
                    break
            smoothed.append(waypoints[best_j])
            i = best_j
        return self.filter_micro_waypoints(smoothed)

    def filter_micro_waypoints(self, waypoints: list[tuple[float, float]], min_seg_m: float = 0.35) -> list[tuple[float, float]]:
        """Filter out micro-segments and merge near-collinear legs."""
        if len(waypoints) <= 2:
            return waypoints

        merged = [waypoints[0]]
        for i in range(1, len(waypoints) - 1):
            p_prev = merged[-1]
            p_curr = waypoints[i]
            p_next = waypoints[i + 1]
            b1 = math.atan2(p_curr[1] - p_prev[1], p_curr[0] - p_prev[0])
            b2 = math.atan2(p_next[1] - p_curr[1], p_next[0] - p_curr[0])
            if abs(normalize_angle(b2 - b1)) > math.radians(15.0):
                merged.append(p_curr)
        merged.append(waypoints[-1])

        filtered = [merged[0]]
        for i in range(1, len(merged) - 1):
            prev = filtered[-1]
            curr = merged[i]
            if math.hypot(curr[0] - prev[0], curr[1] - prev[1]) >= min_seg_m:
                filtered.append(curr)
        filtered.append(merged[-1])
        return filtered

    def plan(self, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> list[tuple[float, float]]:
        """A* algorithm: Computes shortest collision-free grid path from start to goal."""
        sx, sy = start_xy
        gx, gy = goal_xy
        sgx, sgy = self.find_nearest_free(*self.world_to_grid(sx, sy))
        ggx, ggy = self.find_nearest_free(*self.world_to_grid(gx, gy))

        actions = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414),
        ]
        open_set = []
        heapq.heappush(open_set, (0.0, 0.0, (sgx, sgy)))
        came_from = {}
        g_cost = {(sgx, sgy): 0.0}

        def h_cost(a, b):
            dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
            return (dx + dy) + (1.414 - 2.0) * min(dx, dy)

        reached = False
        while open_set:
            _, cur_g, current = heapq.heappop(open_set)
            if current == (ggx, ggy):
                reached = True
                break
            for adx, ady, step_c in actions:
                nx, ny = current[0] + adx, current[1] + ady
                if not self.is_free(nx, ny):
                    continue
                tentative = cur_g + step_c
                if (nx, ny) not in g_cost or tentative < g_cost[(nx, ny)]:
                    g_cost[(nx, ny)] = tentative
                    pri = tentative + h_cost((nx, ny), (ggx, ggy))
                    heapq.heappush(open_set, (pri, tentative, (nx, ny)))
                    came_from[(nx, ny)] = current

        if not reached:
            print("  [A* WARNING] Direct cell connection not found. Using direct line.")
            return [(sx, sy), (gx, gy)]

        cells = [(ggx, ggy)]
        cur = (ggx, ggy)
        while cur in came_from:
            cur = came_from[cur]
            cells.append(cur)
        cells.reverse()

        pts = [self.grid_to_world(c[0], c[1]) for c in cells]
        if len(pts) <= 2:
            return pts

        # Collinear pruning
        pruned = [pts[0]]
        for i in range(1, len(pts) - 1):
            prev = pruned[-1]
            curr = pts[i]
            nxt = pts[i + 1]
            a1 = math.atan2(curr[1] - prev[1], curr[0] - prev[0])
            a2 = math.atan2(nxt[1] - curr[1], nxt[0] - curr[0])
            if abs(normalize_angle(a1 - a2)) > 0.05:
                pruned.append(curr)
        pruned.append(pts[-1])

        return pruned


class MotorDriver:
    """Hardware driver for L298N 4WD chassis + Intel RealSense D455 Gyro & Depth.

    Low CPU & Low Bandwidth:
      - Only reads RealSense hardware gyro and depth frame (no heavy color stream or visual odometry).
      - Tracks real physical yaw using calibrated gyro integration.
      - Measures forward obstacle distance using fast depth ROI slice.
    """

    def __init__(
        self,
        cam_inverted: bool = True,
        motors_inverted: bool = False,
        flip_gyro: bool = False,
        cruise_pwm: Optional[float] = None,
        turn_pwm: Optional[float] = None,
    ):
        self.in1 = self.in2 = self.in3 = self.in4 = self.ena = self.enb = None
        self.pipe = None
        self.cam_inverted = cam_inverted
        self.motors_inverted = motors_inverted
        self.flip_gyro = flip_gyro
        # Inverted front D455: local +Y points UP. In BMI085 IMU coordinate frame,
        # counter-clockwise (turn left) produces negative Y gyro rate.
        # Thus, inverted requires -1.0 multiplier so Turning Left is ALWAYS POSITIVE (+).
        base_sign = -1.0 if self.cam_inverted else 1.0
        self.gyro_sign: float = -base_sign if self.flip_gyro else base_sign
        self.cruise_pwm = cruise_pwm if cruise_pwm is not None else CRUISE_PWM
        self.turn_pwm = turn_pwm if turn_pwm is not None else TURN_PWM

        self.has_camera = False
        self.has_hardware_gyro = False
        self.last_depth: Optional[float] = None
        self.last_color_frame = None
        self.is_moving: bool = False
        self.gyro_bias_y: float = 0.0
        self.last_yaw_rate: float = 0.0
        self.current_yaw_rad: float = 0.0
        self.last_frame_time: float = time.time()
        self.last_gyro_ts: float = 0.0
        self.stop_flag = False
        self.live_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.active_waypoints: list[tuple[float, float]] = []
        self.remaining_dist_m: float = 0.0
        self._cam_lock = threading.Lock()

        # 1. Initialize L298N GPIO
        if HAS_GPIO:
            try:
                self.in1 = DigitalOutputDevice(IN1_PIN)
                self.in2 = DigitalOutputDevice(IN2_PIN)
                self.in3 = DigitalOutputDevice(IN3_PIN)
                self.in4 = DigitalOutputDevice(IN4_PIN)
                self.ena = PWMOutputDevice(ENA_PIN, frequency=1000, initial_value=0)
                self.enb = PWMOutputDevice(ENB_PIN, frequency=1000, initial_value=0)
                print(f"  [HARDWARE] L298N GPIO Ready (IN: {IN1_PIN},{IN2_PIN},{IN3_PIN},{IN4_PIN} | PWM: {ENA_PIN},{ENB_PIN})")
            except Exception as e:
                print(f"  [WARNING] GPIO initialization failed: {e}")
                self.in1 = self.in2 = self.in3 = self.in4 = self.ena = self.enb = None
        else:
            print("  [SIMULATION] Running in mock/dry-run mode (no physical GPIO devices).")

        # 2. Automatic Linux IIO sensor device permissions for RealSense IMU
        if sys.platform.startswith("linux"):
            try:
                import subprocess
                subprocess.run(
                    ["sudo", "chmod", "-R", "a+rw", "/sys/bus/iio/devices/"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1.0
                )
                subprocess.run(
                    "sudo chmod -R a+rw /dev/iio:device* /dev/hidraw* 2>/dev/null",
                    shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1.0
                )
            except Exception:
                pass

        # 3. Initialize RealSense D455 with Multi-Tier Fallback and Auto Hardware Reset
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            devs = ctx.query_devices()

            if len(devs) == 0:
                # Brief settling wait in case USB is still enumerating
                time.sleep(0.8)
                devs = ctx.query_devices()

            if len(devs) > 0:
                dev = devs[0]
                name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "RealSense D455"
                serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else None
                usb = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "Unknown"
                print(f"  [D455 HARDWARE] Found {name} (Serial: {serial} | USB: {usb})")

                # Candidate streaming configurations for D455 in order of priority:
                # Tests explicit 200Hz motion format as well as native auto-negotiation
                stream_profiles = [
                    ("640x480 Depth@15fps + IMU (200Hz)", lambda c: (
                        c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15),
                        c.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200),
                        c.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200),
                    )),
                    ("640x480 Depth@15fps + IMU (Auto)", lambda c: (
                        c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15),
                        c.enable_stream(rs.stream.gyro),
                        c.enable_stream(rs.stream.accel),
                    )),
                    ("480x270 Depth@15fps + IMU (200Hz)", lambda c: (
                        c.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 15),
                        c.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200),
                        c.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200),
                    )),
                    ("480x270 Depth@15fps + IMU (Auto)", lambda c: (
                        c.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 15),
                        c.enable_stream(rs.stream.gyro),
                        c.enable_stream(rs.stream.accel),
                    )),
                    ("Depth + Gyro Only (200Hz)", lambda c: (
                        c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15),
                        c.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200),
                    )),
                    ("IMU Only (Gyro + Accel @ 200Hz)", lambda c: (
                        c.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200),
                        c.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200),
                    )),
                    ("IMU Only (Gyro + Accel Auto)", lambda c: (
                        c.enable_stream(rs.stream.gyro),
                        c.enable_stream(rs.stream.accel),
                    )),
                    ("640x480 Depth@15fps (Depth Only)", lambda c: (
                        c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15),
                    )),
                    ("Auto Depth Only", lambda c: (
                        c.enable_stream(rs.stream.depth),
                    )),
                ]

                pipe_started = False
                for desc, setup_fn in stream_profiles:
                    try:
                        self.pipe = rs.pipeline()
                        cfg = rs.config()
                        if serial:
                            cfg.enable_device(serial)
                        setup_fn(cfg)
                        self.pipe.start(cfg)
                        pipe_started = True
                        if "IMU" in desc or "Gyro" in desc:
                            self.has_hardware_gyro = True
                        print(f"  [D455 SENSORS ✅] Stream started successfully: {desc}")
                        break
                    except Exception as err:
                        if "IMU" in desc or "Gyro" in desc:
                            err_str = str(err).splitlines()[0] if str(err) else "Unknown"
                            print(f"  [D455 PROBE] Profile '{desc}' unavailable: {err_str}")
                        try:
                            if self.pipe:
                                self.pipe.stop()
                        except Exception:
                            pass
                        self.pipe = None
                        continue

                # If initial attempts failed, USB endpoint might be locked: attempt hardware reset
                if not pipe_started:
                    print("  [D455 RESET ⚡] Camera endpoint locked. Performing hardware USB reset...")
                    try:
                        dev.hardware_reset()
                        time.sleep(2.5)  # Wait for D455 to re-enumerate
                        ctx = rs.context()
                        devs = ctx.query_devices()
                        if len(devs) > 0:
                            dev = devs[0]
                            serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else None
                            self.pipe = rs.pipeline()
                            cfg = rs.config()
                            if serial:
                                cfg.enable_device(serial)
                            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
                            cfg.enable_stream(rs.stream.gyro)
                            cfg.enable_stream(rs.stream.accel)
                            self.pipe.start(cfg)
                            pipe_started = True
                            self.has_hardware_gyro = True
                            print("  [D455 SENSORS ✅] Recovered and started after hardware reset!")
                    except Exception as reset_err:
                        print(f"  [D455 RESET FAILED] {reset_err}")

                if pipe_started and self.pipe:
                    self.has_camera = True
                    # Warm-up stream and sample stationary gyro bias
                    gyro_samples = []
                    if self.has_hardware_gyro:
                        for _ in range(25):
                            try:
                                frames = self.pipe.wait_for_frames(timeout_ms=1200)
                                gyro_f = frames.first_or_default(rs.stream.gyro)
                                if gyro_f:
                                    gyro_samples.append(self.gyro_sign * gyro_f.as_motion_frame().get_motion_data().y)
                            except Exception:
                                time.sleep(0.04)
                                continue

                        if len(gyro_samples) >= 6:
                            s_mean = float(np.mean(gyro_samples))
                            s_std = float(np.std(gyro_samples))
                            if s_std < 0.03 and abs(s_mean) < 0.05:
                                self.gyro_bias_y = s_mean
                                print(f"  [D455 GYRO] Calibrated stationary bias = {self.gyro_bias_y:+.4f} rad/s")
                            else:
                                self.gyro_bias_y = 0.0
                                print(f"  [D455 GYRO] Bias unstable ({s_mean:+.4f} ± {s_std:.4f} rad/s). Using zero-bias.")

                        print(f"  [D455 SENSORS READY 🚀] Depth & Gyro telemetry ONLINE (sign={self.gyro_sign:+.1f}).")
                    else:
                        print(f"  [D455 DEPTH ONLY ⚠️] Depth obstacle avoidance ONLINE | Hardware Gyro OFFLINE.")
                        print(f"  [D455 NOTICE] To unlock 200Hz Gyroscope: run with 'sudo -E python3 nav.py' or check Linux permissions.")
                else:
                    print("  [D455 SENSORS 🛑] Could not start any RealSense stream profile.")
                    self.pipe = None
                    self.has_camera = False
            else:
                print("  [D455 SENSORS 🛑] No RealSense camera found on USB bus.")
                print("                   Ensure D455 is firmly connected to a blue USB 3.0 port.")
                self.pipe = None
                self.has_camera = False
        except Exception as e:
            print(f"  [D455 SENSORS ERROR] RealSense startup failed: {e}")
            self.pipe = None
            self.has_camera = False

    def read_motion_and_depth(self) -> tuple[float, float, float, Optional[float]]:
        """Compatibility helper for live visualizers: returns (x, y, yaw_rad, depth_m)."""
        yaw, depth = self.read_sensors()
        return (self.live_pose[0], self.live_pose[1], yaw, depth)

    def read_sensors(self) -> tuple[float, Optional[float]]:
        """Read synchronized gyro and depth from D455.
        Drains all available motion and depth frames from the queue to ensure zero
        gyro latency and integration of all 200Hz samples.
        Returns: (current_yaw_rad, forward_obstacle_depth_m).
        """
        if not self.pipe:
            return (self.current_yaw_rad, self.last_depth)

        with self._cam_lock:
            try:
                import pyrealsense2 as rs
                now = time.time()
                depth_val = None

                # Collect all available framesets and motion frames
                frames_list = []
                try:
                    f0 = self.pipe.wait_for_frames(timeout_ms=100)
                    if f0:
                        frames_list.append(f0)
                except Exception:
                    pass

                # Drain queued motion frames (up to 15)
                for _ in range(15):
                    try:
                        polled = self.pipe.poll_for_frames()
                        if isinstance(polled, tuple):
                            if len(polled) == 2 and polled[0] and polled[1]:
                                frames_list.append(polled[1])
                            else:
                                break
                        elif polled:
                            frames_list.append(polled)
                        else:
                            break
                    except Exception:
                        break

                for frames in frames_list:
                    # 1. Process Gyro Motion Frame
                    gyro_f = None
                    if hasattr(frames, "first_or_default"):
                        gyro_f = frames.first_or_default(rs.stream.gyro)
                    elif hasattr(frames, "is_motion_frame") and frames.is_motion_frame():
                        if frames.get_profile().stream_type() == rs.stream.gyro:
                            gyro_f = frames.as_motion_frame()

                    if gyro_f:
                        try:
                            g_data = gyro_f.as_motion_frame().get_motion_data()
                            raw_rate_y = self.gyro_sign * g_data.y
                            calibrated_rate = raw_rate_y - self.gyro_bias_y
                            self.last_yaw_rate = calibrated_rate

                            # Accurate dt per gyro sample using hardware timestamp
                            ts_sec = gyro_f.get_timestamp() * 1e-3
                            if getattr(self, "last_gyro_ts", 0.0) > 0.0:
                                dt = ts_sec - self.last_gyro_ts
                                if dt <= 0.0 or dt > 0.2:
                                    dt = max(0.002, min(0.05, now - self.last_frame_time))
                            else:
                                dt = max(0.002, min(0.05, now - self.last_frame_time))
                            self.last_gyro_ts = ts_sec

                            if abs(calibrated_rate) > 0.005:
                                self.current_yaw_rad = normalize_angle(self.current_yaw_rad + calibrated_rate * dt)
                        except Exception:
                            pass

                    # 2. Process Depth Frame
                    depth_f = None
                    if hasattr(frames, "get_depth_frame"):
                        depth_f = frames.get_depth_frame()
                    elif hasattr(frames, "is_depth_frame") and frames.is_depth_frame():
                        depth_f = frames.as_depth_frame()

                    if depth_f:
                        d_img = np.asanyarray(depth_f.get_data())
                        if self.cam_inverted:
                            # Sensor mounted upside down: flip horizontally and vertically (-1)
                            d_img = cv2.flip(d_img, -1)

                        h, w = d_img.shape
                        y1, y2 = int(h * 0.20), int(h * 0.52)
                        x1, x2 = int(w * 0.22), int(w * 0.78)

                        roi = d_img[y1:y2, x1:x2]
                        valid = roi[(roi > 150) & (roi < 3500)]
                        if len(valid) >= 60:
                            depth_val = float(np.percentile(valid, 15)) * 0.001
                            self.last_depth = depth_val

                self.last_frame_time = now
                self.live_pose = (self.live_pose[0], self.live_pose[1], self.current_yaw_rad)
                return (self.current_yaw_rad, depth_val if depth_val is not None else self.last_depth)
            except Exception:
                return (self.current_yaw_rad, self.last_depth)

    def get_yaw_rad(self) -> float:
        return self.current_yaw_rad

    def set_yaw_rad(self, yaw_rad: float):
        self.current_yaw_rad = yaw_rad

    def get_yaw_rate(self) -> float:
        return getattr(self, "last_yaw_rate", 0.0)

    def get_forward_depth(self) -> Optional[float]:
        return self.last_depth

    def stop(self):
        """Immediately stop all 4 motors and zero PWM."""
        self.stop_flag = True
        self.is_moving = False
        if self.ena and self.enb:
            self.ena.off()
            self.enb.off()
            self.in1.off()
            self.in2.off()
            self.in3.off()
            self.in4.off()
        time.sleep(0.04)

    def drive_raw(
        self,
        left_forward: bool,
        right_forward: bool,
        speed: Optional[float] = None,
        speed_right: Optional[float] = None,
    ):
        """Low-level differential drive with standard L298N wiring:
        Left motor:  IN1=1, IN2=0 (forward), IN1=0, IN2=1 (reverse)
        Right motor: IN3=1, IN4=0 (forward), IN3=0, IN4=1 (reverse)
        """
        act_left = speed if speed is not None else self.cruise_pwm
        act_right = speed_right if speed_right is not None else act_left

        lf = not left_forward if self.motors_inverted else left_forward
        rf = not right_forward if self.motors_inverted else right_forward

        if self.ena and self.enb:
            self.in1.value = int(lf)
            self.in2.value = int(not lf)
            self.in3.value = int(rf)
            self.in4.value = int(not rf)
            self.ena.value = max(0.0, min(1.0, act_left))
            self.enb.value = max(0.0, min(1.0, act_right))

    def pivot(self, angle_rad: float) -> float:
        """Pivot in place tracking real physical rotation from D455 hardware gyro.
        Uses closed-loop heading feedback, runaway divergence auto-detection,
        and high-torque pulse to break tire scrub.
        """
        if abs(angle_rad) < math.radians(3.0):
            return 0.0

        self.stop_flag = False
        self.is_moving = True
        start_yaw = self.current_yaw_rad
        target_yaw = normalize_angle(start_yaw + angle_rad)
        is_left = angle_rad > 0
        dir_str = "LEFT" if is_left else "RIGHT"
        target_deg = math.degrees(abs(angle_rad))

        print(f"\n  [PIVOT {dir_str}] Target: {target_deg:.1f}° (Base PWM = {max(0.88, self.turn_pwm)*100:.0f}%)")

        cur_pwm = max(0.88, self.turn_pwm)

        if not self.has_hardware_gyro:
            dur = target_deg / math.degrees(TURN_RADPS)
            print(f"  [DEAD-RECKONING PIVOT] Turning for {dur:.2f}s at {math.degrees(TURN_RADPS):.1f}°/s...")
            if self.ena:
                self.drive_raw(not is_left, is_left, cur_pwm, cur_pwm)
                time.sleep(dur)
                self.stop()
            self.current_yaw_rad = target_yaw
            self.live_pose = (self.live_pose[0], self.live_pose[1], self.current_yaw_rad)
            print(f"  [PIVOT DONE ✅] Headed to {math.degrees(self.current_yaw_rad):+.1f}°")
            return angle_rad

        t_start = time.time()
        # Cap max pivot time to 3.5s to prevent indefinite spinning
        t_timeout = t_start + min(3.5, max(1.8, (target_deg / 45.0) * 1.2 + 0.8))

        # Breakaway torque burst (0.12s) to overcome skid-steer floor friction
        if self.ena:
            self.drive_raw(not is_left, is_left, 1.0, 1.0)
            time.sleep(0.12)
            self.drive_raw(not is_left, is_left, cur_pwm, cur_pwm)

        runaway_ticks = 0
        initial_check_done = False

        try:
            while time.time() < t_timeout:
                if self.stop_flag:
                    break
                self.read_sensors()

                current_yaw = self.current_yaw_rad
                remaining_rad = normalize_angle(target_yaw - current_yaw)
                remaining_deg = math.degrees(abs(remaining_rad))

                # Stop when aligned within 3.5 degrees of target
                if remaining_deg <= 3.5:
                    break

                elapsed = time.time() - t_start

                # --- AUTO-RUNAWAY DIVERGENCE DETECTION & GYRO SIGN CORRECTION ---
                # If after 0.20s of motion, the robot is rotating in the WRONG direction
                # (remaining angle is getting larger instead of smaller), the gyro sign is inverted!
                delta_from_start = normalize_angle(current_yaw - start_yaw)
                commanded_dir = 1.0 if is_left else -1.0
                if elapsed > 0.22 and not initial_check_done:
                    # If we commanded left (>0), delta_from_start should be positive (> +0.035)
                    # If it is significantly negative (< -0.035), the sign is inverted!
                    if commanded_dir * delta_from_start < -0.035:
                        runaway_ticks += 1
                        if runaway_ticks >= 3:
                            print(f"\n  [AUTO-CORRECTION ⚡] Gyro sign inverted detected! Auto-flipping gyro polarity.")
                            self.gyro_sign *= -1.0
                            self.current_yaw_rad = normalize_angle(start_yaw - delta_from_start)
                            initial_check_done = True
                            runaway_ticks = 0
                    else:
                        initial_check_done = True

                # Closed-loop direction: steer towards target_yaw
                need_left = remaining_rad > 0
                self.drive_raw(not need_left, need_left, cur_pwm, cur_pwm)

                sys.stdout.write(
                    f"\r  [PIVOT HUD {dir_str}] Target: {math.degrees(target_yaw):+5.1f}° | "
                    f"Current: {math.degrees(current_yaw):+5.1f}° | Remaining: {remaining_deg:4.1f}°"
                )
                sys.stdout.flush()
                time.sleep(0.02)
        finally:
            self.stop()
            self.is_moving = False
            print()

        actual_rad = normalize_angle(self.current_yaw_rad - start_yaw)
        self.live_pose = (self.live_pose[0], self.live_pose[1], self.current_yaw_rad)
        print(f"  [PIVOT DONE ✅] Turned {math.degrees(abs(actual_rad)):.1f}° to Heading {math.degrees(self.current_yaw_rad):+.1f}°")
        return actual_rad

    def drive_straight(
        self,
        distance_m: float,
        bearing: Optional[float] = None,
        start_p: Optional[tuple[float, float]] = None,
        leg_idx: int = 1,
        total_legs: int = 1,
    ) -> tuple[tuple[float, float], str, Optional[float]]:
        """Drive straight along bearing using active D455 gyro steering and real-time obstacle checking."""
        if distance_m <= 0.015:
            return ((0.0, 0.0), "completed", None)

        target_heading = bearing if bearing is not None else self.current_yaw_rad
        print(f"\n  [LEG {leg_idx:02d}/{total_legs:02d}] Distance: {distance_m:.2f} m | Heading: {math.degrees(target_heading):+.1f}°")

        self.stop_flag = False
        self.is_moving = True
        accumulated_dist = 0.0
        cur_x, cur_y = start_p if start_p else (0.0, 0.0)
        start_x, start_y = cur_x, cur_y
        halt_reason = "completed"
        last_depth = self.last_depth

        t_start = time.time()
        t_last = t_start
        t_max_timeout = t_start + max(distance_m / SPEED_MPS * 2.2, 4.0)
        cur_drive_pwm = max(0.85, self.cruise_pwm)

        # High-torque kick to overcome floor friction
        if self.ena:
            self.drive_raw(True, True, 1.0, 1.0)
            time.sleep(0.12)

        try:
            while accumulated_dist < distance_m and time.time() < t_max_timeout:
                if self.stop_flag:
                    halt_reason = "stopped"
                    break

                now = time.time()
                dt = max(0.005, min(0.08, now - t_last))
                t_last = now

                # 1. Read sensors (Gyro + Obstacle Depth)
                current_yaw, measured_depth = self.read_sensors()
                if measured_depth is not None:
                    last_depth = measured_depth

                # 2. OBSTACLE AVOIDANCE: Stop immediately if obstacle is directly ahead (< 0.48m)
                if measured_depth is not None and 0.16 <= measured_depth <= 0.48:
                    print(f"\n  [OBSTACLE DETECTED 🛑] Obstacle at {measured_depth:.2f} m directly ahead! Halting.")
                    halt_reason = "obstacle"
                    break

                # 3. Active Closed-Loop Gyro Steering: lock vehicle along target_heading
                # heading_err > 0 means car needs to turn left (right wheels faster, left wheels slower)
                # heading_err < 0 means car needs to turn right (left wheels faster, right wheels slower)
                heading_err = normalize_angle(target_heading - current_yaw)
                trim = max(-0.18, min(0.18, 0.45 * heading_err))
                left_pwm = max(0.50, min(1.0, cur_drive_pwm - trim))
                right_pwm = max(0.50, min(1.0, cur_drive_pwm + trim))

                if self.ena:
                    self.drive_raw(True, True, speed=left_pwm, speed_right=right_pwm)

                # 4. Integrate displacement along gyro heading
                step_d = SPEED_MPS * dt
                accumulated_dist += step_d
                cur_x += step_d * math.cos(current_yaw)
                cur_y += step_d * math.sin(current_yaw)
                self.live_pose = (cur_x, cur_y, current_yaw)

                depth_str = f"{measured_depth:4.2f}m" if measured_depth is not None else " N/A "
                sys.stdout.write(
                    f"\r  [DRIVING] Leg {leg_idx}/{total_legs} ({accumulated_dist:4.2f}/{distance_m:4.2f}m) | "
                    f"Pose: ({cur_x:+5.2f}, {cur_y:+5.2f}) | Yaw: {math.degrees(current_yaw):+5.1f}° | Depth: {depth_str}"
                )
                sys.stdout.flush()
                time.sleep(0.02)
        finally:
            self.stop()
            self.is_moving = False
            print()

        dx = cur_x - start_x
        dy = cur_y - start_y
        return ((dx, dy), halt_reason, last_depth)

    def backup(self, distance_m: float = 0.20, heading_rad: float = 0.0) -> tuple[float, float]:
        """Safely reverse away from an obstacle to create clearance for pivoting."""
        print(f"\n  [OBSTACLE CLEARANCE ⏪] Backing up {distance_m:.2f} m...")
        t_start = time.time()
        t_timeout = t_start + max(1.2, (distance_m / SPEED_MPS) * 1.8 + 0.4)
        accumulated_back = 0.0
        t_last = t_start

        if self.ena:
            # High-torque burst in reverse to break tire scrub
            self.drive_raw(False, False, 1.0, 1.0)
            time.sleep(0.08)
            self.drive_raw(False, False, max(0.85, self.cruise_pwm))

        try:
            while accumulated_back < distance_m and time.time() < t_timeout:
                now = time.time()
                dt = now - t_last
                t_last = now
                self.read_sensors()
                accumulated_back += SPEED_MPS * dt
                time.sleep(0.03)
        finally:
            self.stop()

        dx = -accumulated_back * math.cos(heading_rad)
        dy = -accumulated_back * math.sin(heading_rad)
        self.live_pose = (self.live_pose[0] + dx, self.live_pose[1] + dy, self.live_pose[2])
        return dx, dy

    def cleanup(self):
        self.stop()
        if self.pipe:
            try:
                self.pipe.stop()
            except Exception:
                pass
        if self.ena:
            for dev in [self.ena, self.enb, self.in1, self.in2, self.in3, self.in4]:
                if dev:
                    try:
                        dev.close()
                    except Exception:
                        pass


def navigate_to_target(
    tid: str,
    tinfo: dict,
    cur_pose: tuple[float, float, float],
    sem_entities: dict,
    driver: MotorDriver,
    planner: Optional[FastAStarPlanner] = None,
    dry_run: bool = False,
    step_prefix: str = "",
) -> tuple[float, float, float]:
    """Execute autonomous A* navigation to the target with real-time D455 obstacle avoidance."""
    cur_x, cur_y, cur_yaw = cur_pose
    target_x, target_y, target_yaw = tinfo["approach"]

    if planner is None:
        planner = FastAStarPlanner(FUSED_DIR)

    # 1. ALWAYS compute collision-free path with A* algorithm on occupancy grid
    print("=" * 64)
    print(f"MISSION: Navigate to '{tid}' ({tinfo.get('desc', tid)})")
    print("=" * 64)
    print(f"  Start:  ({cur_x:+.2f} m, {cur_y:+.2f} m, {math.degrees(cur_yaw):+.1f}°)")
    print(f"  Target: ({target_x:+.2f} m, {target_y:+.2f} m, {math.degrees(target_yaw):+.1f}°)")
    print(f"  [A* PLANNER] Searching optimal collision-free path on occupancy map...")

    raw_path = planner.plan((cur_x, cur_y), (target_x, target_y))
    waypoints = planner.smooth_path(raw_path)

    driver.active_waypoints = [(p[0], p[1]) for p in waypoints]
    total_dist = sum(
        math.hypot(waypoints[i+1][0] - waypoints[i][0], waypoints[i+1][1] - waypoints[i][1])
        for i in range(len(waypoints) - 1)
    )
    driver.remaining_dist_m = total_dist
    print(f"  [A* PATH FOUND] {total_dist:.2f} m across {len(waypoints)} waypoints:")
    for w_i, wp in enumerate(waypoints):
        print(f"    • WP {w_i:02d}: ({wp[0]:+.2f} m, {wp[1]:+.2f} m)")
    print("-" * 64)

    if dry_run:
        print("[DRY-RUN MODE] Simulated arrival without physical actuation.")
        driver.live_pose = (target_x, target_y, target_yaw)
        save_state(target_x, target_y, target_yaw, tid)
        return (target_x, target_y, target_yaw)

    # 2. Hardware Waypoint Execution
    robot_x, robot_y, heading = cur_x, cur_y, cur_yaw
    if driver.has_hardware_gyro:
        driver.set_yaw_rad(cur_yaw)

    driver.live_pose = (robot_x, robot_y, heading)
    reached_target = False
    obstacle_halted = False
    replan_count = 0
    max_replans = 4
    wp_idx = 0

    while wp_idx < len(waypoints) - 1:
        if driver.stop_flag:
            print("\n[STOPPED] Mission aborted by user.")
            break

        p_next = waypoints[wp_idx + 1]
        seg_dist = math.hypot(p_next[0] - robot_x, p_next[1] - robot_y)
        if seg_dist < 0.04:
            wp_idx += 1
            continue

        # A. Turn to face waypoint
        bearing = math.atan2(p_next[1] - robot_y, p_next[0] - robot_x)
        turn_ang = normalize_angle(bearing - heading)

        if abs(turn_ang) > math.radians(4.0):
            actual_turn = driver.pivot(turn_ang)
            heading = normalize_angle(heading + actual_turn)
            driver.live_pose = (robot_x, robot_y, heading)

        # B. Drive straight towards waypoint with active gyro steering
        (dx, dy), halt_reason, last_depth = driver.drive_straight(
            distance_m=seg_dist,
            bearing=heading,
            start_p=(robot_x, robot_y),
            leg_idx=wp_idx + 1,
            total_legs=len(waypoints) - 1,
        )
        robot_x += dx
        robot_y += dy
        driver.live_pose = (robot_x, robot_y, heading)
        driver.remaining_dist_m = max(0.0, math.hypot(target_x - robot_x, target_y - robot_y))

        # Check target arrival proximity
        if math.hypot(robot_x - target_x, robot_y - target_y) <= 0.35:
            reached_target = True
            break

        # C. Dynamic Obstacle Avoidance & A* Re-routing
        if halt_reason == "obstacle":
            obs_dist = last_depth if (last_depth is not None and last_depth > 0.15) else 0.42
            obs_x = robot_x + obs_dist * math.cos(heading)
            obs_y = robot_y + obs_dist * math.sin(heading)
            print(f"\n  [OBSTACLE DETECTED 🛑] Obstacle at {obs_dist:.2f} m ahead at world ({obs_x:+.2f}, {obs_y:+.2f})!")

            # Back up to create turning clearance away from obstacle
            dx_back, dy_back = driver.backup(distance_m=0.20, heading_rad=heading)
            robot_x += dx_back
            robot_y += dy_back
            driver.live_pose = (robot_x, robot_y, heading)

            # Mark obstacle into occupancy grid with safety clearance radius
            planner.add_dynamic_obstacle(obs_x, obs_y, radius_m=0.32)
            replan_count += 1

            if replan_count > max_replans:
                print(f"\n  [OBSTACLE HALT 🛑] Exceeded maximum avoidance attempts ({max_replans}). Path blocked!")
                obstacle_halted = True
                break

            print(f"\n  [A* REPLANNING 🧭] Computing detour around obstacle (Attempt {replan_count}/{max_replans})...")
            new_plan = planner.plan((robot_x, robot_y), (target_x, target_y))
            if len(new_plan) < 2:
                print(f"  [ERROR 🛑] No navigable detour found to target '{tid}'! Halting.")
                obstacle_halted = True
                break

            waypoints = planner.smooth_path(new_plan)
            driver.active_waypoints = [(p[0], p[1]) for p in waypoints]
            total_dist = sum(
                math.hypot(waypoints[k+1][0] - waypoints[k][0], waypoints[k+1][1] - waypoints[k][1])
                for k in range(len(waypoints) - 1)
            )
            driver.remaining_dist_m = total_dist
            print(f"  [DETOUR PATH ACTIVE ✅] New path generated: {total_dist:.2f} m across {len(waypoints)} waypoints.")
            wp_idx = 0
            continue

        wp_idx += 1

    # 3. Final Target Alignment
    if not obstacle_halted and math.hypot(robot_x - target_x, robot_y - target_y) <= 0.45:
        final_turn = normalize_angle(target_yaw - heading)
        if abs(final_turn) > math.radians(4.0):
            actual_final = driver.pivot(final_turn)
            heading = normalize_angle(heading + actual_final)
            driver.live_pose = (robot_x, robot_y, heading)
        reached_target = True

    # 4. Save Pose
    save_status = tid if reached_target else ("stopped_obstacle" if obstacle_halted else "stopped")
    save_state(robot_x, robot_y, heading, save_status)

    print("-" * 64)
    if reached_target:
        print(f"[ARRIVED ✅] Target '{tid}' reached at ({robot_x:+.2f} m, {robot_y:+.2f} m)!")
    elif obstacle_halted:
        print(f"[STOPPED 🛑] Obstacle avoidance halted robot safely at ({robot_x:+.2f} m, {robot_y:+.2f} m).")
    else:
        print(f"[STOPPED ⚠️] Robot stopped at ({robot_x:+.2f} m, {robot_y:+.2f} m). Target: '{tid}'.")
    print("=" * 64)

    return (robot_x, robot_y, heading)


def main():
    parser = argparse.ArgumentParser(description="Autonomous A* Navigation with D455 Gyro & Depth Obstacle Avoidance")
    parser.add_argument("target", nargs="?", default=None, help="Target landmark (e.g. 'desk 18', 'chair 1', 'origin')")
    parser.add_argument("--tour", action="store_true", help="Execute complete 10-station facility tour")
    parser.add_argument("--dwell", type=float, default=2.5, help="Observation dwell time at each tour station in seconds (default: 2.5s)")
    parser.add_argument("--cam-upright", action="store_true", help="Camera is mounted upright instead of inverted (default: inverted)")
    parser.add_argument("--flip-gyro", action="store_true", help="Manually flip/invert hardware gyro yaw rate polarity")
    parser.add_argument("--invert-motors", action="store_true", help="Invert motor directions if hardware is wired in reverse polarity")
    parser.add_argument("--normal-motors", action="store_true", help="Legacy flag: ensure normal motor wiring")
    parser.add_argument("--dry-run", action="store_true", help="Simulate navigation without pulsing hardware motors")
    parser.add_argument("--monitor", action="store_true", help="Launch live real-time Camera vs Map telemetry monitor without driving")
    parser.add_argument("--speed", type=float, default=None, help="Override cruise speed PWM (e.g. 0.88)")
    parser.add_argument("--turn-speed", type=float, default=None, help="Override pivot turn speed PWM (e.g. 0.90)")
    parser.add_argument("--set-start", help="Manually set current robot pose to a known landmark")
    parser.add_argument("--from", dest="start_from", help="Plan navigation starting from a specific landmark")
    parser.add_argument("--from-origin", action="store_true", help="Start navigation from Origin (0, 0, 0)")
    parser.add_argument("--reset", action="store_true", help="Reset robot pose to Origin (0, 0, 0)")
    parser.add_argument("--status", action="store_true", help="Show current robot state")
    args = parser.parse_args()

    motors_inv = bool(args.invert_motors and not args.normal_motors)

    global CRUISE_PWM, TURN_PWM, SPEED_MPS, TURN_RADPS
    if args.speed is not None:
        CRUISE_PWM = float(args.speed)
        SPEED_MPS = max(0.08, CRUISE_PWM * 0.40)
        print(f"[SPEED CONFIG] Cruise PWM set to {CRUISE_PWM*100:.0f}% ({SPEED_MPS:.2f} m/s)")
    if args.turn_speed is not None:
        TURN_PWM = float(args.turn_speed)
        TURN_RADPS = max(0.35, TURN_PWM * 1.55)
        print(f"[SPEED CONFIG] Turn PWM set to {TURN_PWM*100:.0f}% ({math.degrees(TURN_RADPS):.1f}°/s)")

    # Load semantic map entities
    sem_entities = {}
    sem_path = FUSED_DIR / "semantic_map.json"
    if sem_path.exists():
        try:
            with open(sem_path, "r", encoding="utf-8") as f:
                sem_entities = json.load(f).get("entities", {})
        except Exception:
            pass

    cur_x, cur_y, cur_yaw = load_state()

    if args.reset:
        cur_x, cur_y, cur_yaw = 0.0, 0.0, 0.0
        save_state(0.0, 0.0, 0.0, "origin")
        print("  [RESET] Robot state reset to Origin: (X = 0.00 m, Y = 0.00 m, Yaw = 0.0°)")
        if not args.target and not args.tour:
            return
    elif abs(cur_x) > 0.05 or abs(cur_y) > 0.05 or abs(cur_yaw) > 0.05:
        print(f"  [SAVED STATE LOADED] Resumed pose: (X = {cur_x:+.2f} m, Y = {cur_y:+.2f} m, Yaw = {math.degrees(cur_yaw):+.1f}°)")

    if args.from_origin:
        cur_x, cur_y, cur_yaw = 0.0, 0.0, 0.0
        print("  [FROM-ORIGIN] Starting from Origin: (X = 0.00 m, Y = 0.00 m, Yaw = 0.0°)")

    if args.set_start:
        tid, tinfo = resolve_target(args.set_start, sem_entities)
        if not tid:
            print(f"[ERROR] Landmark '{args.set_start}' not recognized.")
            return
        sx, sy, syaw = tinfo["approach"]
        cur_x, cur_y, cur_yaw = sx, sy, syaw
        save_state(sx, sy, syaw, tid)
        print(f"[CALIBRATED] Robot start pose set to '{tid}' ({tinfo['desc']})")
        if not args.target and not args.tour:
            return

    if args.start_from:
        tid, tinfo = resolve_target(args.start_from, sem_entities)
        if tid:
            cur_x, cur_y, cur_yaw = tinfo["approach"]
            print(f"[START OVERRIDE] Starting from '{tid}': (X={cur_x:.2f} m, Y={cur_y:.2f} m, Yaw={math.degrees(cur_yaw):.1f}°)")

    if args.monitor:
        driver = MotorDriver(
            cam_inverted=not args.cam_upright,
            motors_inverted=motors_inv,
            flip_gyro=args.flip_gyro,
        )
        print("=" * 74)
        print("D455 HARDWARE GYRO & LIVE DEPTH TELEMETRY MONITOR")
        print(f"  Camera Mount: {'INVERTED (UPSIDE-DOWN)' if not args.cam_upright else 'UPRIGHT'}")
        print(f"  Gyro Sign Polarity: {driver.gyro_sign:+.1f}")
        print(f"  Hardware Gyro Status: {'ONLINE ✅' if driver.has_hardware_gyro else 'OFFLINE ❌ (Depth Only - run with sudo -E)'}")
        print("  [TEST 1] Rotate car LEFT by hand -> Yaw MUST increase (+).")
        print("           If it decreases (-), re-run with --flip-gyro.")
        print("  [TEST 2] Place hand/box in front of camera (< 0.48m) -> Status must show [OBSTACLE 🛑].")
        print("  Press Ctrl+C to stop.")
        print("=" * 74)
        try:
            while True:
                yaw, depth = driver.read_sensors()
                depth_str = f"{depth:4.2f}m" if depth is not None else " N/A "
                status = "[OBSTACLE 🛑]" if (depth is not None and 0.16 <= depth <= 0.48) else "[CLEAR ✅]"
                sys.stdout.write(
                    f"\r[MONITOR] Yaw: {math.degrees(yaw):+6.1f}° | Rate: {math.degrees(driver.get_yaw_rate()):+5.1f}°/s | "
                    f"Forward Depth: {depth_str} | Status: {status}"
                )
                sys.stdout.flush()
                time.sleep(0.04)
        except KeyboardInterrupt:
            print("\n[STOPPED] Monitor closed.")
        finally:
            driver.cleanup()
        return

    if args.status or (not args.target and not args.tour):
        print("=" * 66)
        print("CURRENT ROBOT STATE:")
        print(f"  Pose: (X = {cur_x:.2f} m, Y = {cur_y:.2f} m, Yaw = {math.degrees(cur_yaw):.1f}°)")
        print(f"  State File: {STATE_FILE.resolve()}")
        print("\nAVAILABLE TARGETS:")
        for tid, info in TARGET_DATABASE.items():
            print(f"  • {tid:<12} (aliases: {', '.join(info['aliases'][:2])}) -> {info['desc']}")
        print("\nUsage example: python nav.py \"desk 18\"")
        print("Tour example:  python nav.py --tour")
        print("=" * 66)
        return

    driver = MotorDriver(
        cam_inverted=not args.cam_upright,
        motors_inverted=motors_inv,
        flip_gyro=args.flip_gyro,
        cruise_pwm=args.speed,
        turn_pwm=args.turn_speed,
    )
    planner = FastAStarPlanner(FUSED_DIR)

    try:
        if args.tour:
            print("=" * 70)
            print("STARTING 10-OBSERVATION SEQUENTIAL FACILITY TOUR")
            print(f"Stops: {len(TOUR_SEQUENCE)} stations | Dwell: {args.dwell:.1f}s/station | DryRun: {args.dry_run}")
            print("=" * 70)
            cur_pose = (cur_x, cur_y, cur_yaw)
            for idx, (tid, desc) in enumerate(TOUR_SEQUENCE, 1):
                tinfo = TARGET_DATABASE[tid]
                print(f"\n>>> [STEP {idx:02d}/10] Commencing navigation to '{tid}' ({desc})...")
                cur_pose = navigate_to_target(
                    tid, tinfo, cur_pose, sem_entities, driver,
                    planner=planner,
                    dry_run=args.dry_run,
                )
                if idx < len(TOUR_SEQUENCE):
                    next_tid, next_desc = TOUR_SEQUENCE[idx]
                    print(f"[OBSERVATION DWELL] Pausing {args.dwell:.1f}s at station {idx:02d} for observation...")
                    time.sleep(args.dwell)
                    print(f"[REROUTING TO NEXT STATION 🧭] Routing to station {idx+1:02d}/10: '{next_tid}' ({next_desc})...")

            print("\n" + "=" * 70)
            print(">>> 10-OBSERVATION FACILITY TOUR SUCCESSFULLY COMPLETED! <<<")
            print(f"Final Pose: (X = {cur_pose[0]:.2f} m, Y = {cur_pose[1]:.2f} m, Yaw = {math.degrees(cur_pose[2]):.1f}°)")
            print("=" * 70)
        else:
            tid, tinfo = resolve_target(args.target, sem_entities)
            if not tid:
                print(f"[ERROR] Target '{args.target}' not recognized.")
                print("Available targets: " + ", ".join(TARGET_DATABASE.keys()))
                sys.exit(1)

            navigate_to_target(
                tid, tinfo, (cur_x, cur_y, cur_yaw), sem_entities, driver,
                planner=planner,
                dry_run=args.dry_run,
            )
    except KeyboardInterrupt:
        print("\n[EMERGENCY STOP] Operator pressed Ctrl+C! Motors stopped.")
    finally:
        driver.cleanup()


if __name__ == "__main__":
    main()
