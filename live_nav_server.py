#!/usr/bin/env python3
"""Live Real-Time Navigation Server & Web Visualizer for Raspberry Pi 5 + Intel RealSense D455.

Features:
  1. Live Low-Latency MJPEG Camera Stream with Real-Time HUD Overlay.
  2. Live Interactive HTML5 Map Visualizer with Real-Time Robot Tracking (X, Y, Yaw).
  3. One-Click Interactive Navigation to any landmark (Desk 1, Desk 18, Chairs, etc.).
  4. One-Click 10-Observation Sequential Facility Tour.
  5. Instant Emergency Stop and Origin Calibration.
  6. Pure Python stdlib (http.server) + OpenCV: Zero external pip dependencies!
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

# Import core navigation and hardware driver from nav.py
import nav
from nav import (
    FastAStarPlanner,
    MotorDriver,
    TARGET_DATABASE,
    TOUR_SEQUENCE,
    FUSED_DIR,
    STATE_FILE,
    load_state,
    save_state,
    resolve_target,
    normalize_angle,
)


class NavigationCoordinator:
    """Thread-safe coordinator managing live telemetry, D455 visual odometry,
    motor actuation, and web visualizer state."""

    def __init__(self, dry_run: bool = False, cam_upright: bool = False):
        self.lock = threading.Lock()
        self.dry_run = dry_run
        self.cam_upright = cam_upright

        # Load planner & entities
        self.planner = FastAStarPlanner(FUSED_DIR)
        self.sem_entities = {}
        sem_path = FUSED_DIR / "semantic_map.json"
        if sem_path.exists():
            try:
                with open(sem_path, "r", encoding="utf-8") as f:
                    self.sem_entities = json.load(f).get("entities", {})
            except Exception:
                pass

        # Load last pose
        cur_x, cur_y, cur_yaw = load_state()
        self.pose = [cur_x, cur_y, cur_yaw]  # [x, y, yaw_rad]
        self.status = "IDLE"
        self.status_detail = "Ready for mission"
        self.target_id = "origin"
        self.target_desc = "Facility Origin"
        self.target_pose = [0.0, 0.0, 0.0]
        self.waypoints: List[List[float]] = []
        self.remaining_dist = 0.0
        self.depth_ahead: Optional[float] = None
        self.tour_active = False
        self.tour_step = 0
        self.emergency_stop_flag = False

        # Pre-render map PNG for web client
        self.map_png_bytes = self._generate_map_png()

        # Initialize hardware
        self.driver = MotorDriver(cam_inverted=not cam_upright)
        self.active_nav_thread: Optional[threading.Thread] = None

        # Start camera frame background loop
        self.running = True
        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()

    def _generate_map_png(self) -> bytes:
        """Render high-contrast, clean 2D occupancy map with semantic entity bounding boxes
        matching master_10_observations_tour_dashboard.png in North-Up orientation.
        """
        # planner.grid has Row 0 at South (Y_min); flip vertically so Row 0 is North (Y_max / top of image)
        grid = np.flipud(self.planner.grid)
        h, w = grid.shape
        rgb = np.ones((h, w, 3), dtype=np.uint8) * 248
        rgb[grid == 205] = [238, 242, 246]  # soft light gray unknown
        rgb[grid == 254] = [255, 255, 255]  # clean white free space
        rgb[grid == 0] = [60, 70, 85]        # dark slate obstacles

        # Overlay all confirmed semantic entities (Desks, Chairs, Boxes) with labels
        for eid, ent in self.sem_entities.items():
            cf = ent.get("centroid_floor")
            dim = ent.get("dimensions", [0.6, 0.6, 0.6])
            cls = ent.get("class_name", "")
            if not cf:
                continue
            ew, el = float(dim[0]), float(dim[1])
            x0, x1 = cf[0] - ew / 2.0, cf[0] + ew / 2.0
            y0, y1 = cf[1] - el / 2.0, cf[1] + el / 2.0

            # Convert to display pixel coordinates where row 0 is North (top)
            px0 = int(round((x0 - self.planner.ox) / self.planner.res))
            px1 = int(round((x1 - self.planner.ox) / self.planner.res))
            py0 = int(round(h - (y1 - self.planner.oy) / self.planner.res))
            py1 = int(round(h - (y0 - self.planner.oy) / self.planner.res))

            px0, px1 = max(0, min(w - 1, px0)), max(0, min(w - 1, px1))
            py0, py1 = max(0, min(h - 1, py0)), max(0, min(h - 1, py1))

            if px1 > px0 and py1 > py0:
                is_tour_target = eid in ("desk_001", "chair_002", "chair_003", "storage box_001", "chair_005", "desk_005", "desk_018", "desk_017", "desk_023", "desk_024")
                fill_color = [219, 234, 254] if "desk" in cls else ([254, 243, 199] if "chair" in cls else [241, 245, 249])
                border_color = (37, 99, 235) if "desk" in cls else ((217, 119, 6) if "chair" in cls else (148, 163, 184))
                
                # Blend translucent fill
                sub = rgb[py0:py1, px0:px1]
                cv2.rectangle(rgb, (px0, py0), (px1, py1), fill_color, -1)
                rgb[py0:py1, px0:px1] = cv2.addWeighted(rgb[py0:py1, px0:px1], 0.55, sub, 0.45, 0)
                cv2.rectangle(rgb, (px0, py0), (px1, py1), border_color, 1)

                if is_tour_target:
                    cv2.putText(rgb, eid, (px0 + 2, max(py0 + 8, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.26, (15, 23, 42), 1, cv2.LINE_AA)

        # Draw subtle border
        cv2.rectangle(rgb, (0, 0), (w - 1, h - 1), (100, 116, 139), 1)
        ok, buf = cv2.imencode(".png", rgb)
        return buf.tobytes() if ok else b""

    def _camera_loop(self):
        """High-frequency background loop reading camera & hardware gyro."""
        while self.running:
            try:
                if self.driver.pipe:
                    if self.status == "IDLE":
                        _, _, depth = self.driver.read_motion_and_depth()
                        with self.lock:
                            self.depth_ahead = depth
                    else:
                        with self.lock:
                            lp = self.driver.live_pose
                            if lp is not None:
                                self.pose = list(lp)
                            self.depth_ahead = self.driver.last_depth
                time.sleep(0.04)
            except Exception:
                time.sleep(0.1)

    def get_latest_jpeg(self) -> bytes:
        """Return annotated live RGB JPEG for MJPEG stream."""
        frame = self.driver.last_color_frame
        if frame is None:
            # Placeholder frame if camera is initializing
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "REAL-TIME D455 CAMERA FEED INITIALIZING...", (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Top telemetry bar
        cv2.rectangle(annotated, (0, 0), (w, 42), (15, 23, 42), -1)
        with self.lock:
            x, y, yaw = self.pose
            yaw_deg = math.degrees(yaw)
            status = self.status
            depth_str = f"{self.depth_ahead:.2f}m" if self.depth_ahead is not None else "--"
            dist_str = f"{self.remaining_dist:.2f}m" if self.remaining_dist > 0 else "0.0m"
            target_str = self.target_id

        # Text on top HUD
        cv2.putText(annotated, f"POSE: ({x:+.2f}, {y:+.2f})m | YAW: {yaw_deg:+5.1f}deg",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"DEPTH: {depth_str} | REMAIN: {dist_str}",
                    (350, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (56, 189, 248), 1, cv2.LINE_AA)

        # Bottom status pill
        status_color = (16, 185, 129) if status in ("IDLE", "ARRIVED") else (56, 189, 248)
        if "OBSTACLE" in status:
            status_color = (239, 68, 68)
        cv2.rectangle(annotated, (12, h - 38), (280, h - 10), (15, 23, 42), -1)
        cv2.putText(annotated, f"STATUS: {status}", (20, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 2, cv2.LINE_AA)

        # Crosshair in center
        cx, cy = w // 2, h // 2
        cv2.line(annotated, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1)
        cv2.line(annotated, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1)

        ok, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return jpeg.tobytes() if ok else b""

    def get_telemetry_json(self) -> dict:
        """Return live telemetry snapshot for Web UI."""
        with self.lock:
            return {
                "pose": {
                    "x": round(self.pose[0], 3),
                    "y": round(self.pose[1], 3),
                    "yaw_rad": round(self.pose[2], 4),
                    "yaw_deg": round(math.degrees(self.pose[2]), 1),
                },
                "status": self.status,
                "status_detail": self.status_detail,
                "target": {
                    "id": self.target_id,
                    "desc": self.target_desc,
                    "x": round(self.target_pose[0], 3),
                    "y": round(self.target_pose[1], 3),
                    "yaw_deg": round(math.degrees(self.target_pose[2]), 1),
                },
                "waypoints": [[round(p[0], 3), round(p[1], 3)] for p in self.waypoints],
                "remaining_dist_m": round(self.remaining_dist, 2),
                "depth_ahead_m": round(self.depth_ahead, 2) if self.depth_ahead is not None else None,
                "tour_active": self.tour_active,
                "tour_step": self.tour_step,
                "timestamp": time.time(),
            }

    def start_navigation(self, target_query: str):
        """Spawn asynchronous navigation thread to target."""
        if self.status not in ("IDLE", "ARRIVED", "STOPPED"):
            return False, "Vehicle is currently busy executing a mission."

        tid, tinfo = resolve_target(target_query, self.sem_entities)
        if not tid:
            return False, f"Target '{target_query}' not recognized in facility database."

        self.emergency_stop_flag = False
        self.tour_active = False

        self.active_nav_thread = threading.Thread(
            target=self._execute_single_mission,
            args=(tid, tinfo),
            daemon=True,
        )
        self.active_nav_thread.start()
        return True, f"Navigating to {tid}"

    def start_navigation_coords(self, x: float, y: float):
        """Navigate to specific (x, y) coordinates on the map."""
        best_tid = None
        best_dist = 0.85
        for tid, info in TARGET_DATABASE.items():
            if tid == "origin":
                continue
            app = info.get("approach", (0, 0, 0))
            d = math.hypot(x - app[0], y - app[1])
            if d < best_dist:
                best_dist = d
                best_tid = tid

        if best_tid:
            return self.start_navigation(best_tid)

        if self.status not in ("IDLE", "ARRIVED", "STOPPED"):
            return False, "Vehicle is currently busy executing a mission."

        cur_x, cur_y, _ = self.pose
        bearing = math.atan2(y - cur_y, x - cur_x)
        custom_info = {
            "aliases": [f"coord_{x:.2f}_{y:.2f}"],
            "approach": (x, y, bearing),
            "center": (x, y),
            "desc": f"Map Point ({x:.2f}m, {y:.2f}m)",
            "frame_idx": 0,
            "timestamp": time.time(),
            "bbox": [100, 100, 400, 300],
        }
        self.emergency_stop_flag = False
        self.tour_active = False
        self.active_nav_thread = threading.Thread(
            target=self._execute_single_mission,
            args=(f"pt_({x:.2f},{y:.2f})", custom_info),
            daemon=True,
        )
        self.active_nav_thread.start()
        return True, f"Navigating to ({x:.2f}m, {y:.2f}m)"

    def start_tour(self):
        """Spawn asynchronous 10-station tour thread."""
        if self.status not in ("IDLE", "ARRIVED", "STOPPED"):
            return False, "Vehicle is currently busy executing a mission."

        self.emergency_stop_flag = False
        self.tour_active = True

        self.active_nav_thread = threading.Thread(
            target=self._execute_tour_mission,
            daemon=True,
        )
        self.active_nav_thread.start()
        return True, "Starting 10-observation facility tour"

    def stop_emergency(self):
        """Immediately halt physical motors and abort active route."""
        self.emergency_stop_flag = True
        self.driver.stop()
        with self.lock:
            self.status = "STOPPED"
            self.status_detail = "EMERGENCY STOPPED BY OPERATOR"
            self.tour_active = False
            self.waypoints = []
            self.remaining_dist = 0.0
        return True, "Motors halted immediately."

    def reset_origin(self):
        """Reset robot coordinates to clean origin."""
        self.emergency_stop_flag = True
        self.driver.stop()
        with self.lock:
            self.pose = [0.0, 0.0, 0.0]
            self.driver.set_yaw_rad(0.0)
            self.target_id = "origin"
            self.target_desc = "Facility Origin"
            self.target_pose = [0.0, 0.0, 0.0]
            self.waypoints = []
            self.remaining_dist = 0.0
            self.status = "IDLE"
            self.status_detail = "Reset to Origin (0.0, 0.0, 0.0)"
        save_state(0.0, 0.0, 0.0, "origin")
        return True, "Robot pose reset to Origin."

    def _execute_single_mission(self, tid: str, tinfo: dict):
        """Worker thread executing single target approach."""
        with self.lock:
            cur_x, cur_y, cur_yaw = self.pose
            target_x, target_y, target_yaw = tinfo["approach"]
            self.target_id = tid
            self.target_desc = tinfo.get("desc", tid)
            self.target_pose = [target_x, target_y, target_yaw]
            self.status = "PLANNING"
            self.status_detail = f"Planning collision-free path to {tid}..."

        # Plan route with 0.20m clearance
        if self.planner.line_of_sight((cur_x, cur_y), (target_x, target_y), clearance_m=0.20):
            wps = [(cur_x, cur_y), (target_x, target_y)]
        else:
            wps = self.planner.plan((cur_x, cur_y), (target_x, target_y))
            wps = self.planner.smooth_path(wps)

        with self.lock:
            self.waypoints = [[p[0], p[1]] for p in wps]
            self.remaining_dist = sum(
                math.hypot(wps[i+1][0] - wps[i][0], wps[i+1][1] - wps[i][1])
                for i in range(len(wps) - 1)
            )

        # Call core navigate_to_target
        try:
            confirmed_pose = nav.navigate_to_target(
                tid=tid,
                tinfo=tinfo,
                cur_pose=(cur_x, cur_y, cur_yaw),
                sem_entities=self.sem_entities,
                driver=self.driver,
                planner=self.planner,
                dry_run=self.dry_run,
            )
            with self.lock:
                self.pose = list(confirmed_pose)
                self.status = "ARRIVED"
                self.status_detail = f"Arrived at {tid} ({tinfo.get('desc', '')})"
                self.remaining_dist = 0.0
        except Exception as e:
            with self.lock:
                self.status = "ERROR"
                self.status_detail = f"Navigation error: {e}"
        finally:
            self.driver.stop()

    def _execute_tour_mission(self):
        """Worker thread executing complete 10-observation sequential facility tour."""
        for idx, (tid, desc) in enumerate(TOUR_SEQUENCE, 1):
            if self.emergency_stop_flag:
                break
            with self.lock:
                self.tour_step = idx
                self.status_detail = f"Tour Step {idx}/10: Heading to {tid} ({desc})"
            tinfo = TARGET_DATABASE[tid]
            with self.lock:
                cur_pose = tuple(self.pose)

            try:
                final_pose = nav.navigate_to_target(
                    tid=tid,
                    tinfo=tinfo,
                    cur_pose=cur_pose,
                    sem_entities=self.sem_entities,
                    driver=self.driver,
                    step_prefix=f"obs_{idx:02d}_{tid}_",
                    planner=self.planner,
                    dry_run=self.dry_run,
                )
                with self.lock:
                    self.pose = list(final_pose)

                if idx < len(TOUR_SEQUENCE) and not self.emergency_stop_flag:
                    with self.lock:
                        self.status = "DWELLING"
                        self.status_detail = f"Station {idx}/10 reached. Dwell capture for 3.0s..."
                    time.sleep(3.0)
            except Exception as e:
                with self.lock:
                    self.status = "ERROR"
                    self.status_detail = f"Tour error at station {idx}: {e}"
                break

        with self.lock:
            if not self.emergency_stop_flag:
                self.status = "ARRIVED"
                self.status_detail = "10-Observation Facility Tour Completed Successfully!"
            self.tour_active = False


# HTML / CSS / JS Single Page Application
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Autonomous D455 Robot Mission Control</title>
  <style>
    :root {
      --bg: #090D16;
      --card-bg: #111827;
      --card-border: #1F2937;
      --accent: #38BDF8;
      --accent-glow: rgba(56, 189, 248, 0.25);
      --success: #10B981;
      --danger: #EF4444;
      --warning: #F59E0B;
      --text-main: #F8FAFC;
      --text-muted: #94A3B8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: rgba(17, 24, 39, 0.85);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--card-border);
      padding: 14px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .badge {
      font-size: 0.75rem;
      padding: 3px 8px;
      border-radius: 999px;
      background: var(--accent-glow);
      color: var(--accent);
      border: 1px solid var(--accent);
      font-weight: 600;
    }
    .status-badge {
      font-size: 0.82rem;
      font-weight: 600;
      padding: 5px 12px;
      border-radius: 6px;
      background: #1E293B;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 8px var(--success);
    }
    .main-grid {
      display: grid;
      grid-template-columns: 1fr 1fr 340px;
      gap: 16px;
      padding: 16px;
      flex: 1;
    }
    @media (max-width: 1200px) {
      .main-grid { grid-template-columns: 1fr 1fr; }
      .sidebar { grid-column: span 2; }
    }
    @media (max-width: 800px) {
      .main-grid { grid-template-columns: 1fr; }
      .sidebar { grid-column: span 1; }
    }
    .panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .panel-header {
      padding: 12px 16px;
      border-bottom: 1px solid var(--card-border);
      font-size: 0.90rem;
      font-weight: 600;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .panel-body {
      flex: 1;
      position: relative;
      background: #0B0F19;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    #camera-feed {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    #map-canvas {
      width: 100%;
      height: 100%;
      cursor: crosshair;
    }
    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .telemetry-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px;
    }
    .telemetry-title {
      font-size: 0.80rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 12px;
      font-weight: 700;
    }
    .telemetry-row {
      display: flex;
      justify-content: space-between;
      margin-bottom: 8px;
      font-size: 0.88rem;
    }
    .telemetry-val {
      font-weight: 600;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      color: var(--accent);
    }
    .btn {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      width: 100%;
      padding: 11px 16px;
      border-radius: 8px;
      border: none;
      font-size: 0.88rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      margin-bottom: 8px;
    }
    .btn-primary {
      background: #0284C7;
      color: white;
    }
    .btn-primary:hover { background: #0369A1; }
    .btn-success {
      background: #059669;
      color: white;
    }
    .btn-success:hover { background: #047857; }
    .btn-danger {
      background: #DC2626;
      color: white;
      font-weight: 700;
    }
    .btn-danger:hover { background: #B91C1C; }
    .btn-secondary {
      background: #1E293B;
      color: var(--text-main);
      border: 1px solid var(--card-border);
    }
    .btn-secondary:hover { background: #334155; }
    .target-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 8px;
    }
    .btn-target {
      padding: 8px 10px;
      font-size: 0.80rem;
      border-radius: 6px;
      background: #1E293B;
      color: #E2E8F0;
      border: 1px solid var(--card-border);
      cursor: pointer;
      text-align: left;
      transition: all 0.15s;
    }
    .btn-target:hover {
      background: #0284C7;
      color: white;
      border-color: #38BDF8;
    }
    .hud-overlay {
      position: absolute;
      top: 12px;
      left: 12px;
      background: rgba(15, 23, 42, 0.80);
      backdrop-filter: blur(6px);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.78rem;
      color: var(--text-main);
      border: 1px solid rgba(255, 255, 255, 0.1);
    .btn:active, .btn-target:active {
      transform: scale(0.96);
    }
    .toast {
      position: fixed;
      top: 68px;
      left: 50%;
      transform: translateX(-50%) translateY(-20px);
      background: rgba(15, 23, 42, 0.96);
      backdrop-filter: blur(10px);
      border: 1px solid var(--accent);
      color: var(--text-main);
      padding: 10px 22px;
      border-radius: 999px;
      font-size: 0.88rem;
      font-weight: 600;
      box-shadow: 0 10px 30px rgba(0,0,0,0.6);
      pointer-events: none;
      opacity: 0;
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      z-index: 9999;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .toast.show {
      transform: translateX(-50%) translateY(0);
      opacity: 1;
    }
  </style>
</head>
<body>
  <div id="toast" class="toast"></div>
  <header>
    <div class="brand">
      <span>🚗</span>
      <span>D455 Autonomous Perception & Navigation</span>
      <span class="badge">RPi 5 Live Engine</span>
    </div>
    <div class="status-badge">
      <div id="status-dot" class="status-dot"></div>
      <span id="header-status">CONNECTING...</span>
    </div>
  </header>

  <div class="main-grid">
    <!-- Panel 1: Live Camera Stream -->
    <div class="panel">
      <div class="panel-header">
        <span>LIVE REALSENSE D455 CAMERA FEED</span>
        <span id="cam-fps" style="font-size: 0.78rem; color: var(--accent);">MJPEG 20 FPS</span>
      </div>
      <div class="panel-body">
        <img id="camera-feed" src="/video_feed" alt="Live D455 Stream">
        <div class="hud-overlay" id="cam-overlay">Depth: -- | Speed: Cruise 55%</div>
      </div>
    </div>

    <!-- Panel 2: Interactive 2D Map Canvas -->
    <div class="panel">
      <div class="panel-header">
        <span>FACILITY OCCUPANCY MAP & TRACED PATH</span>
        <span style="font-size: 0.78rem;">Click landmark to navigate</span>
      </div>
      <div class="panel-body">
        <canvas id="map-canvas"></canvas>
        <div class="hud-overlay" id="map-overlay">Pose: (0.00, 0.00)</div>
      </div>
    </div>

    <!-- Panel 3: Mission Control & Telemetry Sidebar -->
    <div class="sidebar">
      <div class="telemetry-card">
        <div class="telemetry-title">Live Vehicle Telemetry</div>
        <div class="telemetry-row">
          <span>Current X / Y:</span>
          <span class="telemetry-val" id="tel-pos">0.00 m / 0.00 m</span>
        </div>
        <div class="telemetry-row">
          <span>Heading (Yaw):</span>
          <span class="telemetry-val" id="tel-yaw">0.0°</span>
        </div>
        <div class="telemetry-row">
          <span>Forward Depth:</span>
          <span class="telemetry-val" id="tel-depth">-- m</span>
        </div>
        <div class="telemetry-row">
          <span>Target Landmark:</span>
          <span class="telemetry-val" id="tel-target">None</span>
        </div>
        <div class="telemetry-row">
          <span>Path Remaining:</span>
          <span class="telemetry-val" id="tel-dist">0.00 m</span>
        </div>
        <div class="telemetry-row">
          <span>Mission Status:</span>
          <span class="telemetry-val" id="tel-status" style="color: var(--success);">IDLE</span>
        </div>
      </div>

      <div class="telemetry-card">
        <div class="telemetry-title">Mission Operations</div>
        <button class="btn btn-success" onclick="startTour()">▶ Start 10-Station Tour</button>
        <button class="btn btn-secondary" onclick="resetOrigin()">↺ Reset Origin (0,0)</button>
        <button class="btn btn-danger" onclick="emergencyStop()">🛑 EMERGENCY STOP</button>

        <div class="telemetry-title" style="margin-top: 14px;">Direct Landmark Navigation</div>
        <div class="target-grid">
          <button class="btn-target" onclick="navigate('desk 1')">🖥️ Desk 001</button>
          <button class="btn-target" onclick="navigate('chair 2')">🪑 Chair 002</button>
          <button class="btn-target" onclick="navigate('chair 3')">🪑 Chair 003</button>
          <button class="btn-target" onclick="navigate('box 1')">📦 Box 001</button>
          <button class="btn-target" onclick="navigate('chair 5')">🪑 Chair 005</button>
          <button class="btn-target" onclick="navigate('desk 5')">🔬 Desk 005</button>
          <button class="btn-target" onclick="navigate('desk 18')">🖥️ Desk 018</button>
          <button class="btn-target" onclick="navigate('desk 24')">🖥️ Desk 024</button>
        </div>
      </div>
    </div>
  </div>

  <script>
    // --- Live Canvas Map & Telemetry ---
    const canvas = document.getElementById('map-canvas');
    const ctx = canvas.getContext('2d');
    let mapImage = new Image();
    mapImage.src = '/map.png';

    // Map world transform parameters (resolution 0.05m, origin [-7.35, -11.10], size 300x310)
    let mapMeta = {
      ox: -7.35,
      oy: -11.10,
      res: 0.05,
      w: 300,
      h: 310
    };

    let latestTelemetry = null;

    // Load map metadata
    fetch('/api/map_info')
      .then(r => r.json())
      .then(data => {
        mapMeta = data;
        mapImage.onload = () => drawMap();
      });

    function worldToPixel(wx, wy) {
      const px = (wx - mapMeta.ox) / mapMeta.res;
      // Invert Y for HTML5 canvas top-left origin
      const py = mapMeta.h - ((wy - mapMeta.oy) / mapMeta.res);
      return [px, py];
    }

    function drawMap() {
      if (!mapImage.complete) return;
      canvas.width = canvas.parentElement.clientWidth;
      canvas.height = canvas.parentElement.clientHeight;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Fit map image inside canvas maintaining aspect ratio
      const scale = Math.min(canvas.width / mapMeta.w, canvas.height / mapMeta.h) * 0.92;
      const offsetX = (canvas.width - mapMeta.w * scale) / 2;
      const offsetY = (canvas.height - mapMeta.h * scale) / 2;

      ctx.save();
      ctx.translate(offsetX, offsetY);
      ctx.scale(scale, scale);

      // Draw base occupancy map
      ctx.drawImage(mapImage, 0, 0);

      // Draw waypoints path
      if (latestTelemetry && latestTelemetry.waypoints && latestTelemetry.waypoints.length > 1) {
        ctx.strokeStyle = '#0284C7';
        ctx.lineWidth = 3.5 / scale;
        ctx.lineJoin = 'round';
        ctx.beginPath();
        latestTelemetry.waypoints.forEach((wp, idx) => {
          const [px, py] = worldToPixel(wp[0], wp[1]);
          if (idx === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      }

      // Draw Target Approach Halo
      if (latestTelemetry && latestTelemetry.target && latestTelemetry.target.id !== 'origin') {
        const [tx, ty] = worldToPixel(latestTelemetry.target.x, latestTelemetry.target.y);
        ctx.fillStyle = 'rgba(239, 68, 68, 0.4)';
        ctx.strokeStyle = '#EF4444';
        ctx.lineWidth = 2.0 / scale;
        ctx.beginPath();
        ctx.arc(tx, ty, 0.25 / mapMeta.res, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#EF4444';
        ctx.font = `${Math.max(10, 12 / scale)}px sans-serif`;
        ctx.fillText(latestTelemetry.target.id, tx + 6, ty - 6);
      }

      // Draw Robot Avatar (Pulsing Circle + Direction Arrow)
      if (latestTelemetry && latestTelemetry.pose) {
        const [rx, ry] = worldToPixel(latestTelemetry.pose.x, latestTelemetry.pose.y);
        const yaw = latestTelemetry.pose.yaw_rad;

        // Pulsing glow
        ctx.fillStyle = 'rgba(56, 189, 248, 0.35)';
        ctx.beginPath();
        ctx.arc(rx, ry, 0.35 / mapMeta.res, 0, Math.PI * 2);
        ctx.fill();

        // Robot core circle
        ctx.fillStyle = '#38BDF8';
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2.0 / scale;
        ctx.beginPath();
        ctx.arc(rx, ry, 0.18 / mapMeta.res, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Direction pointer (inverted Y for canvas coordinate system)
        const arrowLen = 0.38 / mapMeta.res;
        const dx = Math.cos(yaw) * arrowLen;
        const dy = -Math.sin(yaw) * arrowLen;

        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2.5 / scale;
        ctx.beginPath();
        ctx.moveTo(rx, ry);
        ctx.lineTo(rx + dx, ry + dy);
        ctx.stroke();
      }

      ctx.restore();
    }

    // Toast Notification System
    function showToast(msg, type = 'info') {
      const t = document.getElementById('toast');
      const colors = { info: '#38BDF8', success: '#10B981', error: '#EF4444' };
      t.style.borderColor = colors[type] || colors.info;
      const icon = type === 'error' ? '🛑' : (type === 'success' ? '✅' : 'ℹ️');
      t.innerHTML = `<span>${icon}</span><span>${msg}</span>`;
      t.classList.add('show');
      clearTimeout(window._toastTimeout);
      window._toastTimeout = setTimeout(() => t.classList.remove('show'), 3500);
    }

    // Telemetry Polling Loop (4 Hz = 250ms for smooth UI without server choke)
    let consecutiveFailures = 0;

    function pollTelemetry() {
      fetch('/api/status')
        .then(r => r.json())
        .then(data => {
          consecutiveFailures = 0;
          latestTelemetry = data;
          document.getElementById('header-status').innerText = data.status + (data.tour_active ? ` (${data.tour_step}/10)` : '');
          document.getElementById('status-dot').style.background = (data.status === 'ERROR' || (data.status && data.status.includes('OBSTACLE'))) ? '#EF4444' : '#10B981';
          document.getElementById('tel-status').innerText = data.status;
          document.getElementById('tel-status').style.color = data.status === 'ARRIVED' ? 'var(--success)' : (data.status === 'IDLE' ? 'var(--accent)' : 'var(--warning)');
          document.getElementById('tel-pos').innerText = `${data.pose.x >= 0 ? '+' : ''}${data.pose.x.toFixed(2)}m / ${data.pose.y >= 0 ? '+' : ''}${data.pose.y.toFixed(2)}m`;
          document.getElementById('tel-yaw').innerText = `${data.pose.yaw_deg >= 0 ? '+' : ''}${data.pose.yaw_deg.toFixed(1)}°`;
          document.getElementById('tel-depth').innerText = data.depth_ahead_m !== null ? `${data.depth_ahead_m.toFixed(2)}m` : '--';
          document.getElementById('tel-target').innerText = data.target.id !== 'origin' ? data.target.id : 'None';
          document.getElementById('tel-dist').innerText = `${data.remaining_dist_m.toFixed(2)}m`;

          document.getElementById('map-overlay').innerText = `Pose: (${data.pose.x.toFixed(2)}, ${data.pose.y.toFixed(2)}) | Target: ${data.target.id}`;
          document.getElementById('cam-overlay').innerText = `Depth: ${data.depth_ahead_m !== null ? data.depth_ahead_m.toFixed(2) + 'm' : '--'} | Status: ${data.status}`;

          drawMap();
        })
        .catch(err => {
          consecutiveFailures++;
          if (consecutiveFailures >= 4) {
            document.getElementById('header-status').innerText = 'OFFLINE';
            document.getElementById('status-dot').style.background = '#EF4444';
          }
        });
    }

    setInterval(pollTelemetry, 250);
    window.addEventListener('resize', drawMap);

    // Interactive Command Handlers with Immediate Toast Feedback
    function navigate(target) {
      showToast(`Navigating to ${target}...`, 'info');
      fetch(`/api/navigate?target=${encodeURIComponent(target)}`)
        .then(r => r.json())
        .then(d => showToast(d.message, d.success ? 'success' : 'error'))
        .catch(err => showToast(`Connection error: ${err}`, 'error'));
    }

    function startTour() {
      showToast('Starting 10-Station Facility Tour...', 'info');
      fetch('/api/tour')
        .then(r => r.json())
        .then(d => showToast(d.message, d.success ? 'success' : 'error'))
        .catch(err => showToast(`Connection error: ${err}`, 'error'));
    }

    function resetOrigin() {
      showToast('Resetting pose to Origin (0.00, 0.00, 0.0°)...', 'info');
      fetch('/api/reset')
        .then(r => r.json())
        .then(d => showToast(d.message, d.success ? 'success' : 'error'))
        .catch(err => showToast(`Connection error: ${err}`, 'error'));
    }

    function emergencyStop() {
      fetch('/api/stop')
        .then(r => r.json())
        .then(d => showToast('🛑 EMERGENCY STOP ACTIVATED - Motors Halted', 'error'))
        .catch(err => showToast(`Emergency stop error: ${err}`, 'error'));
    }

    // Interactive Canvas Click-to-Navigate
    canvas.addEventListener('click', (e) => {
      const rect = canvas.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;

      const scale = Math.min(canvas.width / mapMeta.w, canvas.height / mapMeta.h) * 0.92;
      const offsetX = (canvas.width - mapMeta.w * scale) / 2;
      const offsetY = (canvas.height - mapMeta.h * scale) / 2;

      const imgX = (clickX - offsetX) / scale;
      const imgY = (clickY - offsetY) / scale;

      if (imgX >= 0 && imgX < mapMeta.w && imgY >= 0 && imgY < mapMeta.h) {
        const wx = mapMeta.ox + imgX * mapMeta.res;
        const wy = mapMeta.oy + (mapMeta.h - imgY) * mapMeta.res;
        showToast(`Selected map point (${wx.toFixed(2)}m, ${wy.toFixed(2)}m)...`, 'info');
        fetch(`/api/navigate_coords?x=${wx.toFixed(2)}&y=${wy.toFixed(2)}`)
          .then(r => r.json())
          .then(d => showToast(d.message, d.success ? 'success' : 'error'))
          .catch(err => showToast(`Click navigation error: ${err}`, 'error'));
      }
    });
  </script>
</body>
</html>
"""


class WebRequestHandler(BaseHTTPRequestHandler):
    """Custom HTTP handler serving the live web visualizer, MJPEG stream, and API endpoints."""

    coordinator: NavigationCoordinator

    def log_message(self, format, *args):
        # Silence verbose standard logging for clean terminal
        return

    def do_GET(self):
        url = urlparse(self.path)

        if url.path == "/" or url.path == "/index.html":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode("utf-8"))

        elif url.path == "/map.png":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(self.coordinator.map_png_bytes)

        elif url.path == "/video_feed":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while self.coordinator.running:
                    frame = self.coordinator.get_latest_jpeg()
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.05)  # 20 FPS MJPEG stream
            except (ConnectionResetError, BrokenPipeError):
                pass

        elif url.path == "/api/status":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.coordinator.get_telemetry_json()).encode("utf-8"))

        elif url.path == "/api/map_info":
            p = self.coordinator.planner
            info = {
                "ox": p.ox,
                "oy": p.oy,
                "res": p.res,
                "w": p.w,
                "h": p.h,
                "extent": p.extent,
            }
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(info).encode("utf-8"))

        elif url.path == "/api/navigate":
            qs = parse_qs(url.query)
            target = qs.get("target", ["desk 1"])[0]
            ok, msg = self.coordinator.start_navigation(target)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": ok, "message": msg}).encode("utf-8"))

        elif url.path == "/api/navigate_coords":
            qs = parse_qs(url.query)
            try:
                x = float(qs.get("x", [0.0])[0])
                y = float(qs.get("y", [0.0])[0])
                ok, msg = self.coordinator.start_navigation_coords(x, y)
            except Exception as e:
                ok, msg = False, str(e)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": ok, "message": msg}).encode("utf-8"))

        elif url.path == "/api/tour":
            ok, msg = self.coordinator.start_tour()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": ok, "message": msg}).encode("utf-8"))

        elif url.path == "/api/stop":
            ok, msg = self.coordinator.stop_emergency()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": ok, "message": msg}).encode("utf-8"))

        elif url.path == "/api/reset":
            ok, msg = self.coordinator.reset_origin()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": ok, "message": msg}).encode("utf-8"))

        else:
            self.send_error(HTTPStatus.NOT_FOUND)


def main():
    parser = argparse.ArgumentParser(description="Live Web Visualizer & Real-Time Navigation Server")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--cam-upright", action="store_true", help="Camera is mounted upright")
    parser.add_argument("--dry-run", action="store_true", help="Run without physical motor actuation")
    args = parser.parse_args()

    print("=" * 72)
    print("AUTONOMOUS D455 REAL-TIME NAVIGATION SERVER & LIVE WEB VISUALIZER")
    print("=" * 72)
    coordinator = NavigationCoordinator(dry_run=args.dry_run, cam_upright=args.cam_upright)
    WebRequestHandler.coordinator = coordinator
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", args.port), WebRequestHandler)
    print(f"  [WEB VISUALIZER READY] Open in your browser:")
    print(f"  👉 http://localhost:{args.port}  (or http://<PI_IP>:{args.port})")
    print(f"  Live D455 Camera Feed + Interactive Map Visualizer + Telemetry HUD Active.")
    print("=" * 72)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[STOPPING SERVER] Halting motors and closing web visualizer...")
    finally:
        coordinator.running = False
        coordinator.driver.cleanup()
        server.server_close()
        print("[SHUTDOWN] Server cleanly terminated.")


if __name__ == "__main__":
    main()
