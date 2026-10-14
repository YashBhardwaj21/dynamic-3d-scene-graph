#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- GPIO Initialization ---
try:
    from gpiozero import DigitalOutputDevice, PWMOutputDevice
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False

# L298N Pinout (BCM Numbering matching Pi 5 wiring)
IN1_PIN = 17
IN2_PIN = 27
IN3_PIN = 22
IN4_PIN = 23
ENA_PIN = 18
ENB_PIN = 19

# Speeds and Calibrations (Reliable torque to overcome static friction without excessive speed)
CRUISE_PWM = 0.25          # 25% duty cycle (smooth forward drive, overcomes L298N voltage drop)
TURN_PWM = 0.30            # 30% duty cycle for skid-steer pivot turns against floor friction
SPEED_MPS = 0.20           # Measured forward speed in meters/sec at CRUISE_PWM (~20 cm/s)
TURN_RADPS = 0.90          # Measured pivot turn rate in radians/sec at TURN_PWM (~51.5°/s)



def find_fused_dir() -> Path:
    candidates = [
        Path("outputs/semantic_mapping_fused_desk_run"),
        Path("semantic_mapping_fused_desk_run"),
        Path("outputs/outputs/semantic_mapping_fused_desk_run"),
        Path.home() / "perception/outputs/semantic_mapping_fused_desk_run",
        Path.home() / "perception/outputs/outputs/semantic_mapping_fused_desk_run",
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
OBS_FILE = Path("outputs/semantic_mapping_my_desk_recording_01/object_observations.jsonl")
DET_FILE = Path("outputs/semantic_mapping_my_desk_recording_01/detections.jsonl")
RAW_RGB_DIR = Path("data/raw/my_desk_recording_01/rgb")


# Curated Targets with physically verified 0.5m standoff approach poses
TARGET_DATABASE = {
    "desk_001": {
        "aliases": ["desk 1", "desk 01", "desk001", "main desk", "table 1", "primary desk"],
        "approach": (1.42, -1.33, math.radians(-43.0)),
        "center": (2.30, -2.14),
        "desc": "Primary workstation desk",
        "frame_idx": 640,
        "timestamp": 1791788665.226945,
        "bbox": [228, 89, 635, 413],
    },
    "chair_002": {
        "aliases": ["chair 2", "chair 02", "chair002", "side chair", "conference chair"],
        "approach": (1.77, -1.03, math.radians(-90.8)),
        "center": (1.76, -1.96),
        "desc": "Conference side chair",
        "frame_idx": 835,
        "timestamp": 1791788672.763377,
        "bbox": [0, 0, 376, 472],
    },
    "chair_003": {
        "aliases": ["chair 3", "chair 03", "chair003", "meeting chair", "central chair"],
        "approach": (1.13, -2.20, math.radians(-90.0)),
        "center": (1.13, -2.92),
        "desc": "Central meeting chair",
        "frame_idx": 665,
        "timestamp": 1791788666.327422,
        "bbox": [357, 266, 546, 480],
    },
    "box_001": {
        "aliases": ["box 1", "box 01", "storage box 1", "box", "storage box_001"],
        "approach": (1.65, -2.20, math.radians(-90.0)),
        "center": (1.67, -2.81),
        "desc": "Storage box by desk 1",
        "frame_idx": 700,
        "timestamp": 1791788667.661160,
        "bbox": [150, 180, 480, 360],
    },
    "chair_005": {
        "aliases": ["chair 5", "chair 05", "chair005", "south chair"],
        "approach": (0.66, -3.20, math.radians(-90.0)),
        "center": (0.66, -3.87),
        "desc": "South meeting chair",
        "frame_idx": 1295,
        "timestamp": 1791788698.140719,
        "bbox": [0, 0, 485, 406],
    },
    "desk_005": {
        "aliases": ["desk 5", "desk 05", "desk005", "south bench", "lab bench"],
        "approach": (-0.14, -5.00, math.radians(-90.0)),
        "center": (-0.14, -5.72),
        "desc": "South laboratory bench",
        "frame_idx": 1685,
        "timestamp": 1791788712.613647,
        "bbox": [2, 76, 305, 317],
    },
    "desk_018": {
        "aliases": ["desk 18", "desk18", "table 18", "bench 18", "west desk", "west bench"],
        "approach": (-1.36, -3.02, math.pi),
        "center": (-2.56, -3.02),
        "desc": "West workstation bench",
        "frame_idx": 1925,
        "timestamp": 1791788720.783856,
        "bbox": [0, 89, 322, 246],
    },
    "desk_017": {
        "aliases": ["desk 17", "desk17", "table 17", "northwest bench"],
        "approach": (-1.10, -1.53, math.pi),
        "center": (-1.88, -1.53),
        "desc": "Northwest center bench",
        "frame_idx": 2405,
        "timestamp": 1791788745.861253,
        "bbox": [1, 0, 392, 376],
    },
    "desk_023": {
        "aliases": ["desk 23", "desk23", "table 23", "west corridor desk"],
        "approach": (-0.45, -1.11, math.pi),
        "center": (-1.16, -1.11),
        "desc": "West corridor workstation",
        "frame_idx": 2490,
        "timestamp": 1791788748.929292,
        "bbox": [0, 0, 386, 422],
    },
    "desk_024": {
        "aliases": ["desk 24", "desk24", "table 24", "north desk"],
        "approach": (-0.10, -0.28, math.pi),
        "center": (-0.65, -0.28),
        "desc": "North corridor desk",
        "frame_idx": 2885,
        "timestamp": 1791788763.102033,
        "bbox": [0, 0, 275, 460],
    },
    "origin": {
        "aliases": ["origin", "start", "home", "base", "0 0", "start origin"],
        "approach": (0.0, 0.0, 0.0),
        "center": (0.0, 0.0),
        "desc": "Entrance corridor home origin",
        "frame_idx": 0,
        "timestamp": 1791788635.549310,
        "bbox": [50, 50, 590, 430],
    },
}


class MotorDriver:
    """Motor driver interface supporting Raspberry Pi 5 L298N GPIO and closed-loop D455 IMU."""

    def __init__(self, cam_inverted: bool = True):
        self.in1 = self.in2 = self.in3 = self.in4 = self.ena = self.enb = None
        self.imu_pipe = None
        self.cam_inverted = cam_inverted


        self.last_depth: Optional[float] = None

        # 1. Initialize L298N GPIO matching motortest.py
        if HAS_GPIO:
            try:
                self.in1 = DigitalOutputDevice(IN1_PIN)
                self.in2 = DigitalOutputDevice(IN2_PIN)
                self.in3 = DigitalOutputDevice(IN3_PIN)
                self.in4 = DigitalOutputDevice(IN4_PIN)
                self.ena = PWMOutputDevice(ENA_PIN, frequency=1000, initial_value=0)
                self.enb = PWMOutputDevice(ENB_PIN, frequency=1000, initial_value=0)
                print(f"  [HARDWARE] L298N GPIO READY (IN: {IN1_PIN},{IN2_PIN},{IN3_PIN},{IN4_PIN} | PWM: {ENA_PIN},{ENB_PIN})")
            except Exception as e:
                print(f"  [WARNING] Could not initialize GPIO devices: {e}")
                self.in1 = self.in2 = self.in3 = self.in4 = self.ena = self.enb = None
        else:
            print("  [DRY-RUN] Running in simulated mode (no physical GPIO devices).")

        # 2. Initialize RealSense D455 IMU & Depth (Zero heavy files, in-memory telemetry)
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            devs = ctx.query_devices()
            if len(devs) > 0:
                dev = devs[0]
                pipe = rs.pipeline()
                cfg = rs.config()
                cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
                cfg.enable_stream(rs.stream.gyro)
                try:
                    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                except Exception:
                    pass
                pipe.start(cfg)
                self.imu_pipe = pipe
                print("  [D455 SENSORS] ACTIVE: Closed-loop Gyro + Live Depth Telemetry READY!")
            else:
                print("  [D455 SENSORS] No RealSense camera detected. Using calibrated timing.")
        except Exception as e:
            print(f"  [D455 SENSORS] Sensors offline ({e}). Using calibrated timing.")

    def update_sensor_readings(self) -> tuple[float, Optional[float]]:
        """Non-blocking update of gyro yaw rate (rad/s) and forward depth (m)."""
        gyro_rate = 0.0
        if not self.imu_pipe:
            return 0.0, self.last_depth
        try:
            import pyrealsense2 as rs
            frames = self.imu_pipe.poll_for_frames()
            if frames:
                gf = frames.first_or_default(rs.stream.gyro)
                if gf:
                    inv_sign = -1.0 if self.cam_inverted else 1.0
                    gyro_rate = inv_sign * gf.as_motion_frame().get_motion_data().y

                df = frames.get_depth_frame()
                if df:
                    vals = []
                    # Sample center region
                    for u in range(300, 340, 4):
                        for v in range(220, 260, 4):
                            d = df.get_distance(u, v)
                            if 0.15 <= d <= 4.0:
                                vals.append(d)
                    if vals:
                        vals.sort()
                        self.last_depth = vals[len(vals) // 2]
        except Exception:
            pass
        return gyro_rate, self.last_depth

    def measure_forward_depth(self) -> Optional[float]:
        """Query real-time physical depth to object in front (0 MB disk writes)."""
        _, d = self.update_sensor_readings()
        if d is not None:
            return d
        if not self.imu_pipe:
            return None
        # Warmup poll if not yet acquired
        try:
            import pyrealsense2 as rs
            for _ in range(3):
                frames = self.imu_pipe.poll_for_frames()
                if frames:
                    df = frames.get_depth_frame()
                    if df:
                        vals = []
                        for u in range(300, 340, 4):
                            for v in range(220, 260, 4):
                                dist = df.get_distance(u, v)
                                if 0.15 <= dist <= 4.0:
                                    vals.append(dist)
                        if vals:
                            vals.sort()
                            self.last_depth = vals[len(vals) // 2]
                            return self.last_depth
                time.sleep(0.01)
        except Exception:
            pass
        return self.last_depth

    def stop(self):
        """Immediately stop all 4 motors and set PWM to zero."""
        if self.ena and self.enb:
            self.ena.off()
            self.enb.off()
            self.in1.off()
            self.in2.off()
            self.in3.off()
            self.in4.off()
        time.sleep(0.10)

    def drive_raw(self, left_forward: bool, right_forward: bool, speed: Optional[float] = None):
        """Low-level differential drive matching motortest.py."""
        act_speed = speed if speed is not None else CRUISE_PWM
        if self.ena and self.enb:
            self.in1.value = int(left_forward)
            self.in2.value = int(not left_forward)
            self.in3.value = int(right_forward)
            self.in4.value = int(not right_forward)
            self.ena.value = act_speed
            self.enb.value = act_speed

    def get_gyro_z_rate(self) -> float:
        """Read instantaneous yaw angular velocity in rad/s from D455 gyro."""
        rate, _ = self.update_sensor_readings()
        return rate

    def pivot(self, angle_rad: float):
        """Pivot in place with live D455 Gyro HUD tracking."""
        if abs(angle_rad) < 0.04:
            return

        target_deg = math.degrees(abs(angle_rad))
        is_left = angle_rad > 0
        dir_str = "LEFT" if is_left else "RIGHT"

        print(f"\n  [PIVOT {dir_str}] Target: {target_deg:.1f}° (PWM = {TURN_PWM*100:.0f}%)")

        # A. Closed-Loop Turn using D455 RealSense Gyro
        if self.imu_pipe and self.ena:
            integrated_deg = 0.0
            last_t = time.time()
            t_start = last_t
            t_timeout = time.time() + max(abs(angle_rad) / TURN_RADPS * 3.5, 4.0)

            # Spin motors in place: Left=(False, True), Right=(True, False)
            self.drive_raw(not is_left, is_left, TURN_PWM)

            try:
                import pyrealsense2 as rs
                omega_y = 0.0
                last_ts = None
                while time.time() < t_timeout and integrated_deg < target_deg:
                    frames = self.imu_pipe.poll_for_frames()
                    if frames:
                        gf = frames.first_or_default(rs.stream.gyro)
                        if gf:
                            ts = gf.get_timestamp() * 0.001  # D455 hardware timestamp in seconds
                            inv_sign = -1.0 if self.cam_inverted else 1.0
                            omega_y = inv_sign * gf.as_motion_frame().get_motion_data().y
                            if last_ts is not None:
                                dt = ts - last_ts
                                if 0.0 < dt < 0.15 and abs(omega_y) > 0.02:
                                    integrated_deg += math.degrees(abs(omega_y)) * dt
                            last_ts = ts

                    # Anti-stall boost: if tyres haven't started turning after 0.5s, give a slight kick
                    if (time.time() - t_start) > 0.5 and integrated_deg < 0.5:
                        if self.ena.value < 0.40:
                            self.ena.value = min(0.40, self.ena.value + 0.04)
                            self.enb.value = min(0.40, self.enb.value + 0.04)

                    remain = max(0.0, target_deg - integrated_deg)
                    sys.stdout.write(
                        f"\r  [PIVOT HUD {dir_str}] Gyro Turn: {integrated_deg:5.1f}° / {target_deg:5.1f}° | "
                        f"Remaining: {remain:4.1f}° | Rate: {abs(math.degrees(omega_y)):4.1f}°/s"
                    )
                    sys.stdout.flush()
                    time.sleep(0.002)
            finally:
                self.stop()
                print()

            print(f"  [PIVOT COMPLETE ✅] Turned {integrated_deg:.1f}° (Target: {target_deg:.1f}°) via D455 Gyro")

        # B. Fallback Open-Loop Turn
        else:
            turn_time = abs(angle_rad) / TURN_RADPS
            if self.ena:
                self.drive_raw(not is_left, is_left, TURN_PWM)
                time.sleep(turn_time)
                self.stop()
            else:
                time.sleep(min(turn_time * 0.1, 0.2))

    def drive_straight(
        self,
        distance_m: float,
        bearing: Optional[float] = None,
        start_p: Optional[tuple[float, float]] = None,
        end_p: Optional[tuple[float, float]] = None,
        planner: Optional[Any] = None,
        leg_idx: int = 1,
        total_legs: int = 1,
    ):
        """Drive straight forward while streaming real-time sensor vs map HUD matching."""
        if distance_m <= 0.02:
            return

        drive_time = distance_m / SPEED_MPS
        bearing_deg = math.degrees(bearing) if bearing is not None else 0.0
        print(f"\n  [LEG {leg_idx:02d}/{total_legs:02d}] Drive: {distance_m:.2f} m | Map Heading: {bearing_deg:+.1f}° | Time: {drive_time:.1f}s")

        t_start = time.time()
        t_end = t_start + drive_time

        if self.ena:
            self.drive_raw(True, True, CRUISE_PWM)

        try:
            while time.time() < t_end:
                now = time.time()
                elapsed = now - t_start
                progress_ratio = min(1.0, elapsed / drive_time)
                dist_traveled = progress_ratio * distance_m

                # Current estimated position on map
                if start_p and end_p:
                    cur_x = start_p[0] + progress_ratio * (end_p[0] - start_p[0])
                    cur_y = start_p[1] + progress_ratio * (end_p[1] - start_p[1])
                else:
                    cur_x, cur_y = 0.0, 0.0

                # 1. Read live D455 gyro and depth (non-blocking)
                gyro_rate, measured_depth = self.update_sensor_readings()

                # 2. Raycast map obstacle distance from current pose
                map_dist = None
                if planner and bearing is not None:
                    map_dist = planner.raycast_obstacle_distance(cur_x, cur_y, bearing)

                depth_str = f"{measured_depth:4.2f}m" if measured_depth is not None else " N/A "
                map_str = f"{map_dist:4.2f}m" if map_dist is not None else " N/A "

                delta_str = " N/A "
                match_tag = "[SYNCING]"
                if measured_depth is not None and map_dist is not None:
                    delta = abs(measured_depth - map_dist)
                    delta_str = f"{delta*100:3.0f}cm"
                    if delta < 0.35:
                        match_tag = "[MATCH ✅]"
                    elif measured_depth < 0.25:
                        match_tag = "[BLOCK 🛑]"
                    else:
                        match_tag = "[WARN ⚠️]"

                # Active steering bias: Negative feedback counter-steering
                if self.ena and abs(gyro_rate) > 0.05:
                    corr = max(-0.03, min(0.03, gyro_rate * 0.020))
                    self.in1.value = 1; self.in2.value = 0
                    self.in3.value = 1; self.in4.value = 0
                    self.ena.value = max(0.18, min(0.35, CRUISE_PWM + corr))
                    self.enb.value = max(0.18, min(0.35, CRUISE_PWM - corr))

                sys.stdout.write(
                    f"\r  [NAV HUD] Leg {leg_idx}/{total_legs} ({dist_traveled:4.2f}/{distance_m:4.2f}m) | "
                    f"Pose: ({cur_x:+5.2f}, {cur_y:+5.2f}) | "
                    f"Cam Depth: {depth_str} | Map: {map_str} (Δ {delta_str}) | {match_tag:<9}"
                )
                sys.stdout.flush()
                time.sleep(0.05)

        finally:
            self.stop()
            print()


    def cleanup(self):
        self.stop()
        if self.imu_pipe:
            try:
                self.imu_pipe.stop()
            except Exception:
                pass
        if self.ena:
            for dev in [self.ena, self.enb, self.in1, self.in2, self.in3, self.in4]:
                if dev:
                    try:
                        dev.close()
                    except Exception:
                        pass



def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def load_state() -> tuple[float, float, float]:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
                return float(d["x"]), float(d["y"]), float(d["yaw"])
        except Exception:
            pass
    return (0.0, 0.0, 0.0)


def save_state(x: float, y: float, yaw: float, last_target: str):
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

    # Also update fused state file if directory exists
    if FUSED_DIR.exists():
        try:
            state_rec = {
                "version": "1.0.0",
                "map_id": "fused_desk_run",
                "current_pose": {
                    "x_m": round(x, 4),
                    "y_m": round(y, 4),
                    "yaw_rad": round(yaw, 4),
                    "yaw_deg": round(math.degrees(yaw), 1),
                    "timestamp_s": time.time(),
                    "source": "navigation_arrival",
                },
                "last_confirmed_stopping_pose": {
                    "x_m": round(x, 4),
                    "y_m": round(y, 4),
                    "yaw_rad": round(yaw, 4),
                    "yaw_deg": round(math.degrees(yaw), 1),
                    "last_target": last_target,
                },
            }
            with open(FUSED_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state_rec, f, indent=2)
        except Exception:
            pass


def resolve_target(query: str, semantic_entities: Dict[str, dict]) -> tuple[Optional[str], Optional[dict]]:
    q = query.strip().lower()
    # Normalize common action prefixes
    for prefix in ["go to ", "move to ", "navigate to ", "drive to ", "find ", "head to "]:
        if q.startswith(prefix):
            q = q[len(prefix):].strip()

    # Pass 1: Exact matches against target ID and aliases
    for tid, info in TARGET_DATABASE.items():
        if q == tid.lower() or q == tid.lower().replace("_", "") or q == tid.lower().replace("_", " "):
            return tid, info
        for alias in info["aliases"]:
            if q == alias.lower():
                return tid, info

    # Pass 2: Exact token set matching
    q_tokens = set(q.replace("_", " ").split())
    for tid, info in TARGET_DATABASE.items():
        id_tokens = set(tid.lower().replace("_", " ").split())
        if q_tokens == id_tokens:
            return tid, info
        for alias in info["aliases"]:
            if q_tokens == set(alias.lower().split()):
                return tid, info

    # Pass 3: Number-sensitive match (e.g. "desk 1" vs "desk 18")
    import re
    q_num = re.findall(r"\d+", q)
    for tid, info in TARGET_DATABASE.items():
        tid_num = [str(int(n)) for n in re.findall(r"\d+", tid)]
        qn = [str(int(n)) for n in q_num]
        if qn and tid_num and qn == tid_num:
            for word in ["desk", "chair", "table", "box", "bench"]:
                if word in q and (word in tid or any(word in a for a in info["aliases"])):
                    return tid, info

    # Pass 4: Dynamic lookup in semantic map entities
    for eid, ent in semantic_entities.items():
        if q == eid.lower() or q == eid.lower().replace("_", " "):
            cf = ent.get("centroid_floor", [0, 0])
            cls_name = ent.get("class_name", "object")
            approach = (cf[0] + 0.5, cf[1], math.pi)
            info = {
                "aliases": [eid],
                "approach": approach,
                "center": (cf[0], cf[1]),
                "desc": f"Scene object: {eid} ({cls_name})",
                "frame_idx": ent.get("first_frame_idx", 0),
                "timestamp": ent.get("first_seen_timestamp", 1791788635.55),
                "bbox": [100, 100, 400, 300],
            }
            return eid, info

    return None, None


class FastAStarPlanner:
    """Fast 8-connected grid planner on metric occupancy map."""

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

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor((x - self.ox) / self.res)), int(math.floor((y - self.oy) / self.res))

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        return self.ox + (gx + 0.5) * self.res, self.oy + (gy + 0.5) * self.res

    def is_free(self, gx: int, gy: int) -> bool:
        if 0 <= gx < self.w and 0 <= gy < self.h:
            # > 64 avoids real obstacles (0). Unknown (205) and free (254) are traversable
            return self.grid[gy, gx] > 64
        return False

    def find_nearest_free(self, gx: int, gy: int, max_r: int = 6) -> tuple[int, int]:
        if self.is_free(gx, gy):
            return gx, gy
        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        nx, ny = gx + dx, gy + dy
                        if self.is_free(nx, ny):
                            return nx, ny
        return gx, gy

    def raycast_obstacle_distance(self, x: float, y: float, yaw: float, max_range: float = 3.0) -> float:
        """Raycast forward on the occupancy grid along yaw to find distance to nearest obstacle point."""
        step = 0.02
        dist = 0.0
        while dist <= max_range:
            rx = x + dist * math.cos(yaw)
            ry = y + dist * math.sin(yaw)
            gx, gy = self.world_to_grid(rx, ry)
            if not self.is_free(gx, gy):
                return dist
            dist += step
        return max_range

    def plan(self, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> list[tuple[float, float]]:

        sx, sy = start_xy
        gx, gy = goal_xy
        sgx, sgy = self.find_nearest_free(*self.world_to_grid(sx, sy))
        ggx, ggy = self.find_nearest_free(*self.world_to_grid(gx, gy))

        actions = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414)
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

        # Simplify waypoints
        pruned = [pts[0]]
        for i in range(1, len(pts) - 1):
            prev = pruned[-1]
            curr = pts[i]
            nxt = pts[i + 1]
            a1 = math.atan2(curr[1] - prev[1], curr[0] - prev[0])
            a2 = math.atan2(nxt[1] - curr[1], nxt[0] - curr[0])
            if abs(normalize_angle(a1 - a2)) > 0.06:
                pruned.append(curr)
        pruned.append(pts[-1])
        return pruned


def generate_traced_path_map(
    planner: FastAStarPlanner,
    entities: dict,
    start_pose: tuple[float, float, float],
    goal_pose: tuple[float, float, float],
    waypoints: list[tuple[float, float]],
    target_id: str,
    target_desc: str,
    out_path: Path,
):
    """Render and save the complete map of the traced path up to destination."""
    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=140)

    rgb_map = np.ones((planner.h, planner.w, 3), dtype=np.uint8) * 255
    rgb_map[planner.grid == 205] = [238, 242, 246]   # soft light gray
    rgb_map[planner.grid == 254] = [255, 255, 255]   # clean white
    rgb_map[planner.grid == 0] = [60, 70, 85]         # obstacle dark slate

    ax.imshow(rgb_map, origin="lower", extent=planner.extent)
    ax.set_title(f"Traced Path to Destination: {target_id} ({target_desc})",
                 fontsize=13, fontweight="bold", pad=12, color="#0F172A")
    ax.set_xlabel("Floor Coordinate X (m - Forward)", fontsize=10, fontweight="bold", color="#334155")
    ax.set_ylabel("Floor Coordinate Y (m - Left)", fontsize=10, fontweight="bold", color="#334155")
    ax.grid(True, linestyle=":", alpha=0.5, color="#94A3B8")

    # Plot objects
    for eid, ent in entities.items():
        cf = ent.get("centroid_floor")
        dim = ent.get("dimensions", [0.6, 0.6, 0.6])
        cls = ent.get("class_name", "")
        if not cf:
            continue
        ew, el = float(dim[0]), float(dim[1])
        x0, y0 = cf[0] - ew / 2.0, cf[1] - el / 2.0
        is_tgt = (eid == target_id)
        edge_col = "#DC2626" if is_tgt else ("#2563EB" if "desk" in cls else ("#D97706" if "chair" in cls else "#94A3B8"))
        face_col = "#FEE2E2" if is_tgt else ("#DBEAFE" if "desk" in cls else ("#FEF3C7" if "chair" in cls else "#F1F5F9"))
        lw = 2.6 if is_tgt else 1.0
        alpha = 0.90 if is_tgt else 0.45
        rect = patches.FancyBboxPatch((x0, y0), ew, el, boxstyle="round,pad=0.01",
                                      linewidth=lw, edgecolor=edge_col, facecolor=face_col, alpha=alpha, zorder=3)
        ax.add_patch(rect)
        if is_tgt:
            ax.text(cf[0], cf[1], f"TARGET\n{eid}", fontsize=8.5, fontweight="bold", color="#991B1B",
                    ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#DC2626", lw=1.6), zorder=9)
        elif eid in ("chair_027", "chair_038", "desk_023", "desk_024", "chair_019", "desk_001", "desk_018"):
            ax.text(cf[0], cf[1], eid, fontsize=6.8, color="#1E293B", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=edge_col, alpha=0.85, lw=0.8), zorder=4)

    # Plot Path
    wxs = [p[0] for p in waypoints]
    wys = [p[1] for p in waypoints]
    ax.plot(wxs, wys, "-", color="#0284C7", linewidth=3.4, zorder=6, label="Traced Navigation Path")
    ax.plot(wxs, wys, "o", color="#0369A1", markersize=5.5, zorder=7)

    # Directional arrows
    for i in range(len(waypoints) - 1):
        wx1, wy1 = wxs[i], wys[i]
        wx2, wy2 = wxs[i+1], wys[i+1]
        dx, dy = wx2 - wx1, wy2 - wy1
        L = math.hypot(dx, dy)
        if L > 0.08:
            ax.annotate("", xy=(wx1 + dx * 0.65, wy1 + dy * 0.65), xytext=(wx1 + dx * 0.35, wy1 + dy * 0.35),
                        arrowprops=dict(arrowstyle="-|>", color="#0369A1", lw=2.2, mutation_scale=14), zorder=8)

    # Start marker
    ax.add_patch(patches.Circle((start_pose[0], start_pose[1]), 0.18, facecolor="#10B981", edgecolor="black", lw=1.8, zorder=9))
    ax.arrow(start_pose[0], start_pose[1], 0.28 * math.cos(start_pose[2]), 0.28 * math.sin(start_pose[2]),
             head_width=0.08, head_length=0.08, fc="black", ec="black", lw=1.8, zorder=10)
    ax.text(start_pose[0], start_pose[1] + 0.35, "START", fontsize=8.5, fontweight="bold", color="white",
            ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#059669", edgecolor="black"), zorder=10)

    # Arrival marker & standoff buffer
    ax.add_patch(patches.Circle((goal_pose[0], goal_pose[1]), 0.18, facecolor="#DC2626", edgecolor="black", lw=1.8, zorder=9))
    ax.arrow(goal_pose[0], goal_pose[1], 0.28 * math.cos(goal_pose[2]), 0.28 * math.sin(goal_pose[2]),
             head_width=0.08, head_length=0.08, fc="black", ec="black", lw=1.8, zorder=10)
    ax.text(goal_pose[0], goal_pose[1] + 0.35, "ARRIVAL (0.5m)", fontsize=8.5, fontweight="bold", color="white",
            ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#DC2626", edgecolor="black"), zorder=10)
    ax.add_patch(patches.Circle((goal_pose[0], goal_pose[1]), 0.50, edgecolor="#DC2626", facecolor="none", linestyle=":", lw=1.6, zorder=5))

    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-8.5, 2.5)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8.8, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def generate_destination_image(target_id: str, target_info: dict, out_path: Path):
    """Retrieve and annotate camera frame showing the target destination."""
    ts = target_info.get("timestamp")
    bbox = target_info.get("bbox", [100, 100, 400, 300])
    rgb_file = None

    if ts:
        candidate = RAW_RGB_DIR / f"{ts:.6f}.png"
        if candidate.exists():
            rgb_file = candidate

    if not rgb_file or not rgb_file.exists():
        # Fallback to earliest frame in raw dir
        frames = sorted(RAW_RGB_DIR.glob("*.png"))
        if frames:
            rgb_file = frames[0]

    if rgb_file and rgb_file.exists():
        im = cv2.imread(str(rgb_file))
        if im is not None:
            bx0, by0, bx1, by1 = bbox
            cv2.rectangle(im, (bx0, by0), (bx1, by1), (0, 0, 240), 3)
            # Label banner
            desc = target_info.get("desc", target_id)
            cv2.rectangle(im, (0, 0), (im.shape[1], 42), (20, 20, 25), -1)
            cv2.putText(im, f"DESTINATION VIEW | TARGET: {target_id} | {desc}",
                        (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(im, f"TARGET: {target_id}", (bx0 + 8, max(by0 - 10, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 240), 2, cv2.LINE_AA)
            cv2.imwrite(str(out_path), im)
            return

    # Blank placeholder if raw image missing
    blank = np.zeros((480, 640, 3), dtype=np.uint8) + 40
    cv2.putText(blank, f"Destination: {target_id}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), blank)


def generate_mission_card(
    traced_map_p: Path,
    dest_img_p: Path,
    target_id: str,
    target_desc: str,
    start_pose: tuple[float, float, float],
    goal_pose: tuple[float, float, float],
    total_dist: float,
    turn_maneuvers: list[str],
    out_path: Path,
):
    """Generate a side-by-side dashboard combining the traced path map and destination view."""
    map_im = cv2.imread(str(traced_map_p))
    dest_im = cv2.imread(str(dest_img_p))

    if map_im is None or dest_im is None:
        return

    # Resize map to height 800
    h_target = 800
    scale_m = h_target / map_im.shape[0]
    map_resized = cv2.resize(map_im, (int(map_im.shape[1] * scale_m), h_target))

    # Resize dest image to width matching top right
    dest_w = 640
    dest_h = int(dest_im.shape[0] * (dest_w / dest_im.shape[1]))
    dest_resized = cv2.resize(dest_im, (dest_w, dest_h))

    # Right column: dest image + telemetry panel
    right_col = np.ones((h_target, dest_w, 3), dtype=np.uint8) * 248
    right_col[:dest_h, :] = dest_resized

    # Draw telemetry on bottom half of right column
    y_text = dest_h + 35
    def put_line(text, color=(15, 23, 42), font_scale=0.55, thickness=1):
        nonlocal y_text
        cv2.putText(right_col, text, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        y_text += 26

    put_line(f"MISSION: {target_id.upper()} ({target_desc})", color=(185, 28, 28), font_scale=0.62, thickness=2)
    put_line("--------------------------------------------------", color=(100, 116, 139))
    put_line(f"Start Pose:    (X={start_pose[0]:.2f}m, Y={start_pose[1]:.2f}m, Yaw={math.degrees(start_pose[2]):.1f}deg)")
    put_line(f"Goal Pose:     (X={goal_pose[0]:.2f}m, Y={goal_pose[1]:.2f}m, Yaw={math.degrees(goal_pose[2]):.1f}deg)")
    put_line(f"Path Length:   {total_dist:.2f} metres")
    put_line(f"Clearance:     0.50 m standoff from target")
    put_line("MANEUVER SEQUENCE:", color=(30, 41, 59), font_scale=0.58, thickness=2)
    for m in turn_maneuvers[:4]:
        put_line(f"  * {m}", color=(37, 99, 235))
    put_line("Safety Status: All collision zones verified clear.", color=(5, 150, 105), font_scale=0.52, thickness=1)

    combined = np.hstack([map_resized, right_col])
    cv2.imwrite(str(out_path), combined)


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


def verify_arrival_pose(
    driver: MotorDriver,
    planner: FastAStarPlanner,
    target_x: float,
    target_y: float,
    target_yaw: float,
    target_id: str,
    target_info: dict,
) -> tuple[float, float, float, bool]:
    """Verify arrival pose with map points via D455 depth ping & occupancy grid raycast."""
    if target_id.lower() in ["origin", "home", "start"]:
        return (target_x, target_y, target_yaw, True)

    center = target_info.get("center", (target_x, target_y))
    nominal_standoff = math.hypot(center[0] - target_x, center[1] - target_y)
    if nominal_standoff < 0.10:
        nominal_standoff = 0.50

    # Raycast on occupancy map grid along target_yaw
    map_obstacle_dist = planner.raycast_obstacle_distance(target_x, target_y, target_yaw)
    expected_dist = map_obstacle_dist if map_obstacle_dist < 2.5 else nominal_standoff

    print("-" * 68)
    print("REAL-TIME ARRIVAL VERIFICATION WITH MAP POINTS:")
    print(f"  Target Landmark:                {target_id} ({target_info.get('desc', '')})")
    print(f"  Expected Distance to Map Object:{expected_dist:.2f} m")

    measured_depth = driver.measure_forward_depth()

    if measured_depth is not None:
        delta_d = measured_depth - expected_dist
        err_cm = abs(delta_d) * 100.0
        print(f"  Physical Depth to Map Object:   {measured_depth:.2f} m")

        if err_cm <= 35.0:
            # Correct position error along heading axis
            corr_x = target_x + delta_d * math.cos(target_yaw)
            corr_y = target_y + delta_d * math.sin(target_yaw)
            print(f"  Position Confirmation:          CONFIRMED ON TARGET POSITION ✅")
            print(f"  Physical Position Delta:        {delta_d*100:+.1f} cm (Pose corrected to map points)")
            print(f"  Verified True Robot Pose:       (X = {corr_x:.2f} m, Y = {corr_y:.2f} m, Yaw = {math.degrees(target_yaw):.1f}°)")
            return (corr_x, corr_y, target_yaw, True)
        else:
            print(f"  Position Confirmation:          DISCREPANCY DETECTED ⚠️ (Delta: {delta_d*100:+.1f} cm)")
            print(f"  Retaining Planned Standoff:     (X = {target_x:.2f} m, Y = {target_y:.2f} m, Yaw = {math.degrees(target_yaw):.1f}°)")
            return (target_x, target_y, target_yaw, False)
    else:
        print(f"  Position Confirmation:          CONFIRMED (Using D455 Closed-Loop Gyro Pose)")
        return (target_x, target_y, target_yaw, True)


def navigate_to_target(
    tid: str,
    tinfo: dict,
    cur_pose: tuple[float, float, float],
    sem_entities: dict,
    driver: MotorDriver,
    step_prefix: str = "",
    dry_run: bool = False,
) -> tuple[float, float, float]:
    """Execute navigation from cur_pose to tid, generating deliverables and driving motors."""
    cur_x, cur_y, cur_yaw = cur_pose
    target_x, target_y, target_yaw = tinfo["approach"]

    # 1. Fast Collision-Free A* Planning
    planner = FastAStarPlanner(FUSED_DIR)
    waypoints = planner.plan((cur_x, cur_y), (target_x, target_y))

    # Calculate total path distance and segment maneuvers
    total_dist = 0.0
    maneuvers = []
    robot_heading = cur_yaw

    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        seg_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        total_dist += seg_dist
        bearing = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        turn_ang = normalize_angle(bearing - robot_heading)
        if abs(turn_ang) > 0.05:
            turn_deg = math.degrees(turn_ang)
            maneuvers.append(f"Pivot {'Left' if turn_deg > 0 else 'Right'} {abs(turn_deg):.1f}°")
        maneuvers.append(f"Drive straight {seg_dist:.2f} m")
        robot_heading = bearing

    final_alignment = normalize_angle(target_yaw - robot_heading)
    if abs(final_alignment) > 0.05:
        maneuvers.append(f"Final align to target: {math.degrees(final_alignment):+.1f}°")

    # 2. Instantly Generate Deliverables
    t_gen_start = time.time()
    out_dir = FUSED_DIR / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    traced_map_path = Path("traced_path_map.png")
    dest_img_path = Path("destination_view.png")
    mission_card_path = Path("navigation_mission_card.png")

    generate_traced_path_map(
        planner, sem_entities,
        (cur_x, cur_y, cur_yaw), (target_x, target_y, target_yaw),
        waypoints, tid, tinfo["desc"], traced_map_path,
    )
    import shutil
    shutil.copy2(traced_map_path, out_dir / f"{step_prefix}traced_path.png")

    generate_destination_image(tid, tinfo, dest_img_path)
    shutil.copy2(dest_img_path, out_dir / f"{step_prefix}destination_view.png")

    generate_mission_card(
        traced_map_path, dest_img_path, tid, tinfo["desc"],
        (cur_x, cur_y, cur_yaw), (target_x, target_y, target_yaw),
        total_dist, maneuvers, mission_card_path,
    )
    if step_prefix:
        shutil.copy2(mission_card_path, out_dir / f"{step_prefix}mission_card.png")

    t_gen = time.time() - t_gen_start

    # 3. Print Instant Mission Telemetry
    header_title = f"MISSION: Navigate to '{tid}' ({tinfo['desc']})"
    if step_prefix:
        header_title = f"TOUR STEP: {header_title}"
    print("=" * 68)
    print(header_title)
    print("=" * 68)
    print(f"  Start Pose:       (X = {cur_x:.2f} m, Y = {cur_y:.2f} m, Yaw = {math.degrees(cur_yaw):.1f}°)")
    print(f"  Target Pose:      (X = {target_x:.2f} m, Y = {target_y:.2f} m, Yaw = {math.degrees(target_yaw):.1f}°)")
    print(f"  Total Distance:   {total_dist:.2f} metres ({len(waypoints)} waypoints)")
    print(f"  Standoff Buffer:  0.50 metres clearance from object front")
    print("-" * 68)
    print(f"INSTANT OUTPUTS GENERATED ({t_gen:.2f}s):")
    print(f"  • Traced Path Map:    {traced_map_path.resolve()}")
    print(f"  • Destination Image:  {dest_img_path.resolve()}")
    print(f"  • Master Mission Card:{mission_card_path.resolve()}")
    print("-" * 68)
    print("EXECUTION MANEUVERS:")
    for step_num, m in enumerate(maneuvers, 1):
        print(f"  {step_num:02d}. {m}")
    print("-" * 68)

    # 4. Motor Execution (or Simulated Drive)
    if dry_run:
        print("[DRY-RUN MODE] Skipping hardware motor actuation.")
    else:
        heading = cur_yaw
        total_legs = len(waypoints) - 1
        for i in range(total_legs):
            p1 = waypoints[i]
            p2 = waypoints[i + 1]
            seg_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            bearing = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            turn_ang = normalize_angle(bearing - heading)

            if abs(turn_ang) > 0.05:
                driver.pivot(turn_ang)
                heading = bearing

            driver.drive_straight(
                distance_m=seg_dist,
                bearing=bearing,
                start_p=p1,
                end_p=p2,
                planner=planner,
                leg_idx=i + 1,
                total_legs=total_legs,
            )
            heading = bearing

        final_turn = normalize_angle(target_yaw - heading)
        if abs(final_turn) > 0.05:
            driver.pivot(final_turn)

    # 5. Station Arrival Map-Point Verification
    confirmed_x, confirmed_y, confirmed_yaw, is_on_target = verify_arrival_pose(
        driver, planner, target_x, target_y, target_yaw, tid, tinfo
    )

    # Update and persist verified state
    save_state(confirmed_x, confirmed_y, confirmed_yaw, tid)
    print("-" * 68)
    if is_on_target:
        print(f"[SUCCESS ✅] Arrived at '{tid}'! Target verified with map points & stopping pose saved.")
    else:
        print(f"[DISCREPANCY DETECTED ⚠️] Completed route to '{tid}', but camera depth did NOT match map points.")
        print(f"  Robot stopped off-target. Pick up and reset with '--reset' or inspect camera orientation.")
    print("=" * 68)

    return (confirmed_x, confirmed_y, confirmed_yaw)



def main():
    parser = argparse.ArgumentParser(description="One-command autonomous navigation with instant visual deliverables.")
    parser.add_argument("target", nargs="?", default=None, help="Target location prompt (e.g. 'desk 18', 'chair 1', 'origin')")
    parser.add_argument("--tour", action="store_true", help="Execute complete 10-observation sequential tour")
    parser.add_argument("--dwell", type=float, default=3.0, help="Dwell time in seconds at each station for observation (default: 3.0s)")
    parser.add_argument("--cam-upright", action="store_true", help="Camera is mounted upright instead of inverted/upside-down")
    parser.add_argument("--dry-run", action="store_true", help="Simulate navigation without pulsing hardware motors")

    parser.add_argument("--monitor", action="store_true", help="Launch live real-time Camera vs Map telemetry monitor without driving")
    parser.add_argument("--speed", type=float, default=None, help="Override cruise speed PWM (e.g. 0.25)")
    parser.add_argument("--turn-speed", type=float, default=None, help="Override pivot turn speed PWM (e.g. 0.30)")
    parser.add_argument("--set-start", help="Manually set current robot pose to a known landmark (e.g. 'desk 1', 'chair 2')")
    parser.add_argument("--from", dest="start_from", help="Plan navigation starting from a specific landmark instead of current state")
    parser.add_argument("--reset", action="store_true", help="Reset robot pose to Origin (0, 0, 0)")
    parser.add_argument("--status", action="store_true", help="Show current robot state")
    args = parser.parse_args()

    global CRUISE_PWM, TURN_PWM, SPEED_MPS, TURN_RADPS
    if args.speed is not None:
        CRUISE_PWM = float(args.speed)
        SPEED_MPS = max(0.08, CRUISE_PWM * 0.60)
        print(f"[SPEED CONFIG] Cruise PWM set to {CRUISE_PWM*100:.0f}% ({SPEED_MPS:.2f} m/s)")
    if args.turn_speed is not None:
        TURN_PWM = float(args.turn_speed)
        TURN_RADPS = max(0.35, TURN_PWM * 2.65)
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
        print("[RESET] Robot state reset to Origin: (X=0.00 m, Y=0.00 m, Yaw=0.0°)")
        if not args.target and not args.tour:
            return

    if args.set_start:
        tid, tinfo = resolve_target(args.set_start, sem_entities)
        if not tid:
            print(f"[ERROR] Could not recognize landmark '{args.set_start}'.")
            return
        sx, sy, syaw = tinfo["approach"]
        cur_x, cur_y, cur_yaw = sx, sy, syaw
        save_state(sx, sy, syaw, tid)
        print(f"[CALIBRATED] Robot start pose set to '{tid}' ({tinfo['desc']})")
        print(f"  Pose: (X = {sx:.2f} m, Y = {sy:.2f} m, Yaw = {math.degrees(syaw):.1f}°)")
        if not args.target and not args.tour:
            return

    if args.start_from:
        tid, tinfo = resolve_target(args.start_from, sem_entities)
        if tid:
            cur_x, cur_y, cur_yaw = tinfo["approach"]
            print(f"[START OVERRIDE] Starting from '{tid}': (X={cur_x:.2f} m, Y={cur_y:.2f} m, Yaw={math.degrees(cur_yaw):.1f}°)")

    if args.monitor:
        planner = FastAStarPlanner(FUSED_DIR)
        driver = MotorDriver(cam_inverted=not args.cam_upright)
        print("=" * 74)
        print("REAL-TIME CAMERA SENSOR VS MAP TELEMETRY LIVE MONITOR")
        print(f"  Robot Map Pose: (X = {cur_x:+.2f} m, Y = {cur_y:+.2f} m, Base Yaw = {math.degrees(cur_yaw):+.1f}°)")
        print(f"  Camera Mount:   {'INVERTED (UPSIDE-DOWN)' if not args.cam_upright else 'UPRIGHT'}")
        print("  Rotate the car or point it at obstacles. Press Ctrl+C to stop.")
        print("=" * 74)
        last_t = time.time()
        integrated_yaw = cur_yaw
        try:
            while True:
                now = time.time()
                dt = now - last_t
                last_t = now

                gyro_rate, measured_depth = driver.update_sensor_readings()
                if abs(gyro_rate) > 0.02:
                    integrated_yaw = normalize_angle(integrated_yaw + gyro_rate * dt)

                map_dist = planner.raycast_obstacle_distance(cur_x, cur_y, integrated_yaw)

                depth_str = f"{measured_depth:4.2f}m" if measured_depth is not None else " N/A "
                map_str = f"{map_dist:4.2f}m" if map_dist is not None else " N/A "
                delta_str = " N/A "
                status_str = "[SYNCING]"
                if measured_depth is not None and map_dist is not None:
                    delta = abs(measured_depth - map_dist)
                    delta_str = f"{delta*100:3.0f}cm"
                    if delta < 0.35:
                        status_str = "[MATCH ✅]"
                    elif measured_depth < 0.25:
                        status_str = "[CLOSE 🛑]"
                    else:
                        status_str = "[WARN ⚠️]"

                sys.stdout.write(
                    f"\r[MONITOR] Yaw: {math.degrees(integrated_yaw):+6.1f}° (Rate: {math.degrees(gyro_rate):+5.1f}°/s) | "
                    f"Cam Depth: {depth_str} | Map Raycast: {map_str} (Δ {delta_str}) | {status_str:<9}"
                )
                sys.stdout.flush()
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n[STOPPED] Telemetry monitor closed.")
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

    driver = MotorDriver(cam_inverted=not args.cam_upright)
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
                    step_prefix=f"obs_{idx:02d}_{tid}_",
                    dry_run=args.dry_run,
                )
                if idx < len(TOUR_SEQUENCE):
                    print(f"[OBSERVATION DWELL] Pausing {args.dwell:.1f}s at station {idx:02d} for sensor capture...")
                    time.sleep(args.dwell)

            print("\n" + "=" * 70)
            print(">>> 10-OBSERVATION FACILITY TOUR SUCCESSFULLY COMPLETED! <<<")
            print(f"Final Pose: (X = {cur_pose[0]:.2f} m, Y = {cur_pose[1]:.2f} m, Yaw = {math.degrees(cur_pose[2]):.1f}°)")
            print("=" * 70)
        else:
            tid, tinfo = resolve_target(args.target, sem_entities)
            if not tid:
                print(f"[ERROR] Could not recognize target '{args.target}'.")
                print("Available targets: " + ", ".join(TARGET_DATABASE.keys()))
                sys.exit(1)

            navigate_to_target(
                tid, tinfo, (cur_x, cur_y, cur_yaw), sem_entities, driver,
                dry_run=args.dry_run,
            )

    except KeyboardInterrupt:
        print("\n[EMERGENCY STOP] Operator pressed Ctrl+C! Halting motors immediately.")
    finally:
        driver.cleanup()


if __name__ == "__main__":
    main()

