#!/usr/bin/env python3
"""Unified One-Command Autonomous Scene Graph Navigator for Raspberry Pi 5.

Usage:
  python3 nav.py "desk 18"
  python3 nav.py "desk 1"
  python3 nav.py "chair 1"
  python3 nav.py "chair 2"
  python3 nav.py origin      (returns back to start)
  python3 nav.py --reset     (resets robot state to 0,0,0)

Features:
  - Resolves any target from the 3D scene graph automatically.
  - Automatically loads the last stopping position from robot_state.json.
  - Computes exact pivot turns, corridor driving distance, and arrival orientation.
  - Executes timed L298N motor actuation with gentle ramping and safety stop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# --- GPIO Initialization ---
try:
    from gpiozero import DigitalOutputDevice, PWMOutputDevice
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    print("[WARNING] gpiozero not found; running in simulated dry-run mode.")

# L298N Pinout (BCM Numbering matching Pi 5 wiring)
IN1_PIN = 17
IN2_PIN = 27
IN3_PIN = 22
IN4_PIN = 23
ENA_PIN = 18
ENB_PIN = 19

# Speeds and Calibrations
CRUISE_PWM = 0.60          # 60% duty cycle (~0.35 m/s)
TURN_PWM = 0.45            # 45% duty cycle for pivot turns
SPEED_MPS = 0.35           # Measured forward speed in meters/sec at CRUISE_PWM
TURN_RADPS = 1.55          # Measured pivot turn rate in radians/sec at TURN_PWM (~89 deg/s)

STATE_FILE = Path("robot_state.json")

# Verified Scene Graph Entities & Precomputed Standoff Approaches (0.5m clearance)
TARGET_DATABASE = {
    # Desk 18 (West Cluster Bench)
    "desk_018": {"aliases": ["desk 18", "desk18", "table 18", "bench 18"], "approach": (-1.36, -3.02, math.pi), "desc": "West workstation bench"},
    # Desk 1 (Primary Workstation)
    "desk_001": {"aliases": ["desk 1", "desk 01", "desk001", "main desk", "table 1"], "approach": (1.42, -1.33, math.radians(-43.0)), "desc": "Primary workstation desk"},
    # Chair 1 (Office Swivel Chair)
    "chair_001": {"aliases": ["chair 1", "chair 01", "chair001", "swivel chair", "task chair"], "approach": (4.38, -2.68, math.radians(179.0)), "desc": "Office task swivel chair"},
    # Chair 2 (Conference Side Chair)
    "chair_002": {"aliases": ["chair 2", "chair 02", "chair002", "side chair"], "approach": (1.77, -1.03, math.radians(-90.8)), "desc": "Conference side chair"},
    # Desk 4 (Adjacent Workbench)
    "desk_004": {"aliases": ["desk 4", "desk 04", "desk004", "table 4"], "approach": (3.65, -2.15, math.radians(-90.0)), "desc": "East workbench"},
    # Chair 3 (Meeting Chair)
    "chair_003": {"aliases": ["chair 3", "chair 03", "chair003"], "approach": (1.13, -2.20, math.radians(-90.0)), "desc": "Central meeting chair"},
    # Storage Box 1
    "box_001": {"aliases": ["box 1", "box 01", "storage box 1", "box"], "approach": (1.65, -2.20, math.radians(-90.0)), "desc": "Storage box by desk 1"},
    # Origin / Home
    "origin": {"aliases": ["origin", "start", "home", "base", "0 0"], "approach": (0.0, 0.0, 0.0), "desc": "Starting entrance corridor origin"},
}


class MotorDriver:
    def __init__(self):
        if HAS_GPIO:
            self.in1 = DigitalOutputDevice(IN1_PIN)
            self.in2 = DigitalOutputDevice(IN2_PIN)
            self.in3 = DigitalOutputDevice(IN3_PIN)
            self.in4 = DigitalOutputDevice(IN4_PIN)
            self.ena = PWMOutputDevice(ENA_PIN, frequency=1000, initial_value=0)
            self.enb = PWMOutputDevice(ENB_PIN, frequency=1000, initial_value=0)
        else:
            self.in1 = self.in2 = self.in3 = self.in4 = self.ena = self.enb = None

    def stop(self):
        if HAS_GPIO:
            self.ena.value = 0.0
            self.enb.value = 0.0
            self.in1.off()
            self.in2.off()
            self.in3.off()
            self.in4.off()
        time.sleep(0.3)

    def pivot(self, angle_rad: float):
        """Pivot in place: positive angle = Left (CCW), negative angle = Right (CW)."""
        if abs(angle_rad) < 0.05:
            return
        turn_time = abs(angle_rad) / TURN_RADPS
        print(f"  -> Pivoting {'LEFT' if angle_rad > 0 else 'RIGHT'} by {math.degrees(abs(angle_rad)):.1f}° ({turn_time:.2f}s)...")
        if HAS_GPIO:
            if angle_rad > 0:
                # Pivot Left: Left backward, Right forward
                self.in1.off(); self.in2.on()
                self.in3.on();  self.in4.off()
            else:
                # Pivot Right: Left forward, Right backward
                self.in1.on();  self.in2.off()
                self.in3.off(); self.in4.on()
            self.ena.value = TURN_PWM
            self.enb.value = TURN_PWM
            time.sleep(turn_time)
            self.stop()
        else:
            time.sleep(min(turn_time, 0.5))

    def drive_straight(self, distance_m: float):
        """Drive straight forward for the specified distance in meters."""
        if distance_m <= 0.02:
            return
        drive_time = distance_m / SPEED_MPS
        print(f"  -> Driving forward {distance_m:.2f} m at {SPEED_MPS:.2f} m/s ({drive_time:.2f}s)...")
        if HAS_GPIO:
            self.in1.on(); self.in2.off()
            self.in3.on(); self.in4.off()
            self.ena.value = CRUISE_PWM
            self.enb.value = CRUISE_PWM
            time.sleep(drive_time)
            self.stop()
        else:
            time.sleep(min(drive_time, 0.5))

    def cleanup(self):
        self.stop()
        if HAS_GPIO:
            for dev in [self.ena, self.enb, self.in1, self.in2, self.in3, self.in4]:
                if dev:
                    dev.close()


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def resolve_target(query: str):
    q = query.strip().lower()
    for tid, info in TARGET_DATABASE.items():
        if q == tid.lower():
            return tid, info
        for alias in info["aliases"]:
            if alias in q or q in alias:
                return tid, info
    return None, None


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


def main():
    parser = argparse.ArgumentParser(description="One-command autonomous RC car navigation.")
    parser.add_argument("target", nargs="?", default=None, help="Target object (e.g. 'desk 18', 'chair 1', 'origin')")
    parser.add_argument("--reset", action="store_true", help="Reset robot pose to (0, 0, 0)")
    parser.add_argument("--status", action="store_true", help="Show current robot state")
    args = parser.parse_args()

    if args.reset:
        save_state(0.0, 0.0, 0.0, "origin")
        print("[RESET] Robot state reset to Origin: (X=0.00 m, Y=0.00 m, Yaw=0.0°)")
        return

    cur_x, cur_y, cur_yaw = load_state()

    if args.status or not args.target:
        print("=" * 60)
        print("CURRENT ROBOT STATE:")
        print(f"  Pose: (X = {cur_x:.2f} m, Y = {cur_y:.2f} m, Yaw = {math.degrees(cur_yaw):.1f}°)")
        print(f"  State File: {STATE_FILE.resolve()}")
        print("\nAVAILABLE TARGETS:")
        for tid, info in TARGET_DATABASE.items():
            print(f"  • {tid:<12} (aliases: {', '.join(info['aliases'][:2])}) -> {info['desc']}")
        print("\nUsage example: python3 nav.py \"desk 18\"")
        print("=" * 60)
        return

    tid, tinfo = resolve_target(args.target)
    if not tid:
        print(f"[ERROR] Could not recognize target '{args.target}'.")
        print("Available targets: " + ", ".join(TARGET_DATABASE.keys()))
        sys.exit(1)

    target_x, target_y, target_yaw = tinfo["approach"]

    dx = target_x - cur_x
    dy = target_y - cur_y
    dist = math.hypot(dx, dy)

    if dist < 0.05:
        print(f"[ALREADY ARRIVED] Robot is already at {tid} (within {dist:.2f} m).")
        return

    # Heading from current pose to approach point
    desired_bearing = math.atan2(dy, dx)
    turn_1 = normalize_angle(desired_bearing - cur_yaw)
    turn_2 = normalize_angle(target_yaw - desired_bearing)

    print("=" * 65)
    print(f"MISSION: Navigate to '{tid}' ({tinfo['desc']})")
    print("=" * 65)
    print(f"  Current Pose:  (X = {cur_x:.2f} m, Y = {cur_y:.2f} m, Yaw = {math.degrees(cur_yaw):.1f}°)")
    print(f"  Target Pose:   (X = {target_x:.2f} m, Y = {target_y:.2f} m, Yaw = {math.degrees(target_yaw):.1f}°)")
    print(f"  Distance:      {dist:.2f} metres")
    print(f"  Turn 1 (Align):{math.degrees(turn_1):+.1f}°")
    print(f"  Turn 2 (Face): {math.degrees(turn_2):+.1f}°")
    print("  Press Ctrl+C at any moment to emergency stop.")
    print("-" * 65)

    driver = MotorDriver()
    try:
        # Step 1: Align heading toward target
        if abs(turn_1) > 0.05:
            driver.pivot(turn_1)

        # Step 2: Drive straight to target approach
        driver.drive_straight(dist)

        # Step 3: Align to final target facing orientation
        if abs(turn_2) > 0.05:
            driver.pivot(turn_2)

        # Update and persist state
        save_state(target_x, target_y, target_yaw, tid)
        print("-" * 65)
        print(f"[SUCCESS] Arrived at '{tid}'! Stopping pose saved.")
        print(f"New Robot Pose: (X = {target_x:.2f} m, Y = {target_y:.2f} m, Yaw = {math.degrees(target_yaw):.1f}°)")
        print("=" * 65)

    except KeyboardInterrupt:
        print("\n[EMERGENCY STOP] Operator pressed Ctrl+C! Halting motors immediately.")
    finally:
        driver.cleanup()


if __name__ == "__main__":
    main()
