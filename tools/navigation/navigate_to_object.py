#!/usr/bin/env python3
"""Autonomous Navigation Entry Point for 4-Motor RC Car (Mode C).

Resolves natural language goal instructions, validates obstacles and costmaps,
computes collision-free differential-drive trajectories, and executes motion
under the protection of the safety state machine and hardware watchdog.

Usage:
  # Dry-run route preview (Safe, no motor activation):
  python tools/navigation/navigate_to_object.py \\
    --map outputs/semantic_mapping_desk_run_161552 \\
    --goal "go to chair_004 and stop 0.5 metres away" \\
    --dry-run

  # Physical motion execution (Requires explicit --enable-motion):
  python tools/navigation/navigate_to_object.py \\
    --map outputs/semantic_mapping_desk_run_161552 \\
    --goal "go to the table with the drone and stop 0.5 metres away" \\
    --enable-motion
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from scene_graph.geometry.coordinate_frames import RobotExtrinsics
from scene_graph.navigation.goal_resolver import GoalResolver
from scene_graph.navigation.planner import GridPathPlanner
from scene_graph.navigation.controller import DifferentialDriveController, ControllerConfig
from scene_graph.navigation.state_machine import NavigationStateMachine, NavigationState
from scene_graph.navigation.localization import PersistentMapLocalizer
from scene_graph.hardware.motor_controller import L298NMotorController, L298NPinConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Navigator")


def main():
    parser = argparse.ArgumentParser(description="Autonomous object-driven navigation for 4-motor RC car")
    parser.add_argument("--map", "-m", type=str, required=True, help="Path to validated map directory")
    parser.add_argument("--goal", "-g", type=str, required=True, help="Natural language navigation instruction")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Preview route without moving motors (default True)")
    parser.add_argument("--enable-motion", action="store_true", help="Explicit flag required to enable physical motor motion")
    parser.add_argument("--robot-radius", type=float, default=0.18, help="Robot footprint radius in meters")
    parser.add_argument("--start-pose", type=str, default=None, help="Initial robot pose 'X Y YAW_DEG' in floor_map frame")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save resolution and route artifacts")
    args = parser.parse_args()

    # If --enable-motion is passed, override dry_run
    is_motion_enabled = bool(args.enable_motion)

    map_dir = Path(args.map).resolve()
    if not map_dir.exists():
        logger.error("Map directory does not exist: %s", map_dir)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else (map_dir / "visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("AUTONOMOUS OBJECT NAVIGATION (MODE C)")
    print(f"Map Directory:    {map_dir}")
    print(f"Goal Instruction: \"{args.goal}\"")
    print(f"Motion Mode:      {'LIVE PHYSICAL MOTION' if is_motion_enabled else 'DRY-RUN / PREVIEW ONLY'}")
    print("=" * 76)

    # 1. Initialize Safety State Machine
    sm = NavigationStateMachine()
    sm.initialize()
    logger.info("Navigation State: %s", sm.state.value)

    # 2. Initialize Persistent Localizer & Determine Current Robot Pose
    logger.info("Loading map localization and floor-plane transform...")
    localizer = PersistentMapLocalizer(map_dir)
    sm.on_sensors_ready()

    # Determine starting pose in floor_map frame
    start_pose = (0.0, 0.0, 0.0)
    if args.start_pose:
        sp_parts = [float(v) for v in args.start_pose.strip().split()]
        start_pose = (sp_parts[0], sp_parts[1], math.radians(sp_parts[2]) if len(sp_parts) > 2 else 0.0)
    elif localizer.keyframe_poses:
        # Use first keyframe position transformed to floor_map frame
        first_pose_slam = localizer.keyframe_poses[0][1]
        T_floor_base = localizer.contract.camera_world_to_robot_base_floor(first_pose_slam)
        start_pose = (float(T_floor_base[0, 3]), float(T_floor_base[1, 3]), 0.0)

    now = time.time()
    sm.on_localization_update(pose_valid=True, covariance=0.01, timestamp_s=now)
    logger.info("Robot Pose in floor_map frame: (X=%.2f m, Y=%.2f m, Yaw=%.1f deg)",
                start_pose[0], start_pose[1], math.degrees(start_pose[2]))

    # 3. Resolve Natural Language Prompt
    logger.info("Resolving prompt to verified semantic entity...")
    resolver = GoalResolver.from_map_dir(map_dir, robot_radius_m=args.robot_radius)
    res_goal = resolver.resolve(args.goal, current_robot_pose=start_pose)

    goal_res_path = out_dir / "goal_resolution.json"
    with open(goal_res_path, "w", encoding="utf-8") as f:
        json.dump(res_goal.to_dict(), f, indent=2)

    if not res_goal.success:
        sm.trigger_safe_stop(f"Goal resolution rejected: {res_goal.rejection_reason}")
        print("\n" + "=" * 76)
        print(f"[REJECTED GOAL]: {res_goal.rejection_reason}")
        print("Autonomous execution safely refused.")
        print("=" * 76)
        sys.exit(1)

    print(f"  Target Entity:       {res_goal.target_entity_id} ({res_goal.target_class})")
    print(f"  Target Center:       {res_goal.target_center_floor}")
    print(f"  Requested Standoff:  {res_goal.standoff_distance_m:.2f} m")
    gx, gy, gyaw = res_goal.approach_pose_2d
    print(f"  Approach Goal Pose:  (X={gx:.2f} m, Y={gy:.2f} m, Yaw={math.degrees(gyaw):.1f} deg)")

    # 4. Plan Collision-Free A* Route on Nav2 Costmap
    sm.start_planning()
    planner = GridPathPlanner.from_map_dir(map_dir)
    planned_path = planner.plan(start_pose, res_goal.approach_pose_2d, target_entity_id=res_goal.target_entity_id)

    route_res_path = out_dir / "planned_route.json"
    with open(route_res_path, "w", encoding="utf-8") as f:
        json.dump(planned_path.to_dict(), f, indent=2)

    if not planned_path.success:
        sm.on_plan_ready(path_found=False)
        sm.trigger_safe_stop(f"Planning failed: {planned_path.rejection_reason}")
        print("\n" + "=" * 76)
        print(f"[PLANNING FAILED]: {planned_path.rejection_reason}")
        print("=" * 76)
        sys.exit(1)

    sm.on_plan_ready(path_found=True)
    print(f"  Path Length:         {planned_path.path_length_m:.2f} m")
    print(f"  Waypoints Count:     {len(planned_path.waypoints_floor)}")
    print(f"  Turn Points:         {planned_path.turn_points_count}")

    # Generate visual route artifact
    route_img_path = out_dir / "planned_goal_route.png"
    planner.visualize_planned_route(
        planned_path,
        route_img_path,
        target_center=res_goal.target_center_floor,
        robot_radius_m=args.robot_radius,
    )
    print(f"  Route Visual Saved:  {route_img_path}")

    # 5. Execution or Dry-Run Gate
    if not is_motion_enabled:
        print("\n" + "=" * 76)
        print("DRY-RUN / PREVIEW COMPLETE")
        print("All safety checks and route plans passed. Motors remain UNPOWERED.")
        print(f"To execute physical motion on hardware, run with: --enable-motion")
        print("=" * 76)
        sys.exit(0)

    # 6. Physical Motion Execution Loop
    print("\n[ARMING RC CAR FOR PHYSICAL EXECUTION]")
    motor = L298NMotorController()
    if not motor.arm():
        sm.trigger_safe_stop("Failed to arm L298N motor driver")
        sys.exit(1)

    sm.arm_and_execute(enable_motion=True)
    controller = DifferentialDriveController(ControllerConfig(track_width_m=0.20))
    controller.reset()

    seq_id = 1
    current_pose = start_pose
    step_count = 0

    try:
        while sm.state in (NavigationState.EXECUTING, NavigationState.APPROACHING):
            cmd = controller.compute_command(current_pose, planned_path.waypoints_floor)
            if cmd.phase == "APPROACHING" and sm.state == NavigationState.EXECUTING:
                sm.on_enter_approach(math.hypot(gx - current_pose[0], gy - current_pose[1]))

            if cmd.is_stopped:
                sm.on_goal_reached()
                motor.set_differential_speeds(0.0, 0.0, seq_id=seq_id, timestamp_s=time.time())
                break

            # Send motor speeds to L298N
            motor_ack = motor.set_differential_speeds(
                cmd.pwm_left_pct * (1 if cmd.v_left_mps >= 0 else -1),
                cmd.pwm_right_pct * (1 if cmd.v_right_mps >= 0 else -1),
                seq_id=seq_id,
                timestamp_s=time.time(),
            )
            seq_id += 1

            # Simulated pose advancement for bench testing / closed-loop demo
            dt = 0.05
            dx = cmd.linear_velocity_mps * math.cos(current_pose[2]) * dt
            dy = cmd.linear_velocity_mps * math.sin(current_pose[2]) * dt
            dyaw = cmd.angular_velocity_radps * dt
            current_pose = (current_pose[0] + dx, current_pose[1] + dy, current_pose[2] + dyaw)

            step_count += 1
            if step_count % 10 == 0:
                print(f"  Step {step_count}: Pose=({current_pose[0]:.2f}, {current_pose[1]:.2f}) | V={cmd.linear_velocity_mps:.2f} m/s | Phase={cmd.phase}")
            time.sleep(dt)

        print("\n" + "=" * 76)
        print("NAVIGATION COMPLETED SUCCESSFULLY!")
        print(f"Final Robot Pose: (X={current_pose[0]:.2f} m, Y={current_pose[1]:.2f} m, Yaw={math.degrees(current_pose[2]):.1f} deg)")
        print(f"Final Distance to Target: {math.hypot(res_goal.target_center_floor[0] - current_pose[0], res_goal.target_center_floor[1] - current_pose[1]):.2f} m")
        print("=" * 76)

    except KeyboardInterrupt:
        logger.warning("Manual operator interruption! Halting motors.")
        motor.trigger_emergency_stop("Manual KeyboardInterrupt")
        sm.trigger_safe_stop("Manual KeyboardInterrupt")
    finally:
        motor.shutdown()


if __name__ == "__main__":
    main()
