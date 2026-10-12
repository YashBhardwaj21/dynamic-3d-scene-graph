#!/usr/bin/env python3
"""Autonomous Navigation Entry Point for 4-Motor RC Car (Mode C).

Resolves natural language goal instructions, validates obstacles and costmaps,
computes collision-free differential-drive trajectories, executes motion
under the protection of the safety state machine, maintains persistent state across
sequential commands, and executes live obstacle-avoidance replanning.

Usage:
  # Sequential Command 1 (Dry-run preview):
  python tools/navigation/navigate_to_object.py \
    --map outputs/semantic_mapping_fused_desk_run \
    --start-pose "0.0 0.0 0.0" \
    --goal "go to desk_001 and stop 0.5 metres away" \
    --dry-run

  # Sequential Command 2 (Automatically uses saved pose from Command 1):
  python tools/navigation/navigate_to_object.py \
    --map outputs/semantic_mapping_fused_desk_run \
    --goal "go to chair_001 and stop 0.5 metres away" \
    --simulate-motion

  # Sequential Command 3 (Automatically uses saved pose from Command 2):
  python tools/navigation/navigate_to_object.py \
    --map outputs/semantic_mapping_fused_desk_run \
    --goal "go to door_001 and stop 0.5 metres away" \
    --simulate-motion
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import numpy as np
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
from scene_graph.navigation.state_manager import NavigationStateManager, PoseSource, CommandStatus
from scene_graph.navigation.obstacle_monitor import LiveObstacleMonitor, ObstacleSafetyStatus
from scene_graph.hardware.motor_controller import L298NMotorController, L298NPinConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Navigator")


def _point_to_polyline_distance(px: float, py: float, waypoints: list) -> float:
    if not waypoints:
        return 0.0
    if len(waypoints) == 1:
        return math.hypot(px - waypoints[0][0], py - waypoints[0][1])

    min_dist = float("inf")
    for i in range(len(waypoints) - 1):
        ax, ay = waypoints[i][0], waypoints[i][1]
        bx, by = waypoints[i + 1][0], waypoints[i + 1][1]
        dx = bx - ax
        dy = by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-6:
            dist = math.hypot(px - ax, py - ay)
        else:
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
            proj_x = ax + t * dx
            proj_y = ay + t * dy
            dist = math.hypot(px - proj_x, py - proj_y)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def main():
    parser = argparse.ArgumentParser(description="Autonomous object-driven navigation for 4-motor RC car")
    parser.add_argument("--map", "-m", type=str, required=True, help="Path to validated map directory")
    parser.add_argument("--goal", "-g", type=str, required=True, help="Natural language navigation instruction")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Preview route without moving motors (default True)")
    parser.add_argument("--enable-motion", action="store_true", help="Explicit flag required to enable physical motor motion")
    parser.add_argument("--simulate-motion", action="store_true", help="Enable kinematic simulated motion updates (bench/test mode only)")
    parser.add_argument("--robot-radius", type=float, default=0.18, help="Robot footprint radius in meters")
    parser.add_argument("--start-pose", type=str, default=None, help="Explicit initial robot pose 'X Y YAW_DEG' in floor_map frame")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save resolution and route artifacts")
    parser.add_argument("--allow-unknown", action="store_true", help="Permit navigation across unobserved space (default False)")
    parser.add_argument("--allow-goal-snap", action="store_true", help="Snap approach pose to nearest free cell if slightly inside costmap inflation")
    parser.add_argument("--max-endpoint-adj", type=float, default=0.10, help="Maximum allowed snap adjustment between resolved goal and grid endpoint in meters")
    parser.add_argument("--state-file", type=str, default=None, help="Path to persistent robot navigation state JSON")
    parser.add_argument("--reset-state", action="store_true", help="Clear persistent state and restart fresh")
    parser.add_argument("--simulate-obstacle-step", type=int, default=None, help="Simulate a dynamic obstacle appearing at control step N")
    parser.add_argument("--simulate-obstacle-dist", type=float, default=0.32, help="Distance of simulated dynamic obstacle in meters")
    parser.add_argument("--allow-open-loop-motion", action="store_true", help="Allow physical motor actuation on Raspberry Pi without external live camera localization")
    args = parser.parse_args()

    is_motion_enabled = bool(args.enable_motion or args.simulate_motion)

    map_dir = Path(args.map).resolve()
    if not map_dir.exists():
        logger.error("Map directory does not exist: %s", map_dir)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else (map_dir / "visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize State Manager
    state_file = Path(args.state_file).resolve() if args.state_file else (map_dir / "robot_navigation_state.json")
    if args.reset_state and state_file.exists():
        try:
            state_file.unlink()
            logger.info("Cleared previous navigation state file: %s", state_file)
        except Exception:
            pass

    state_mgr = NavigationStateManager(map_dir, state_file=state_file)

    print("=" * 76)
    print("AUTONOMOUS OBJECT NAVIGATION (MODE C)")
    print(f"Map Directory:    {map_dir}")
    print(f"Goal Instruction: \"{args.goal}\"")
    print(f"Motion Mode:      {'LIVE PHYSICAL MOTION' if args.enable_motion else ('SIMULATED MOTION' if args.simulate_motion else 'DRY-RUN / PREVIEW ONLY')}")
    print(f"State File:       {state_mgr.state_file}")
    print("=" * 76)

    # 2. Initialize Safety State Machine & Localizer
    sm = NavigationStateMachine()
    sm.initialize()
    localizer = PersistentMapLocalizer(map_dir)
    sm.on_sensors_ready()

    start_pose = None
    start_pose_source = PoseSource.INVALID
    is_live_localization = False

    # Resolve Start Pose: User CLI -> Persisted State -> Live Sensor Localization
    if args.start_pose:
        sp_parts = [float(v) for v in args.start_pose.strip().split()]
        start_pose = (sp_parts[0], sp_parts[1], math.radians(sp_parts[2]) if len(sp_parts) > 2 else 0.0)
        start_pose_source = PoseSource.MANUAL_INITIALIZED
        localizer.set_manual_pose(start_pose)
        state_mgr.update_pose(start_pose, source=PoseSource.MANUAL_INITIALIZED)
        logger.info("[USER_SPECIFIED START POSE]: (X=%.2f m, Y=%.2f m, Yaw=%.1f deg)",
                    start_pose[0], start_pose[1], math.degrees(start_pose[2]))
        sm.on_localization_update(pose_valid=True, covariance=0.001, timestamp_s=time.time())
    else:
        # Check Persistent State
        persisted_pose = state_mgr.get_initial_pose_for_planning(
            allow_command_estimated=(args.simulate_motion or not is_motion_enabled or args.allow_open_loop_motion),
            allow_historical_checkpoint=True,
        )

        if persisted_pose is not None:
            start_pose = persisted_pose
            rec = state_mgr.current_pose or state_mgr.last_confirmed_stopping_pose
            start_pose_source = rec.source if rec else PoseSource.HISTORICAL_CHECKPOINT
            localizer.set_manual_pose(start_pose)
            logger.info("[RECOVERED PERSISTENT POSE]: (X=%.2f m, Y=%.2f m, Yaw=%.1f deg) Source=%s",
                        start_pose[0], start_pose[1], math.degrees(start_pose[2]), start_pose_source.value)
            sm.on_localization_update(pose_valid=True, covariance=0.005, timestamp_s=time.time())
        else:
            # Require live measured localization estimate from localizer
            loc_est = localizer.get_current_pose(max_stale_duration_s=0.50)
            if loc_est.is_valid and loc_est.status in ("TRACKED", "DEGRADED") and loc_est.covariance_norm < 0.10:
                start_pose = loc_est.robot_pose_2d
                start_pose_source = PoseSource.LOCALIZED
                is_live_localization = True
                sm.on_localization_update(pose_valid=True, covariance=loc_est.covariance_norm, timestamp_s=loc_est.timestamp_s)
                logger.info("[MEASURED LIVE POSE]: (X=%.2f m, Y=%.2f m, Yaw=%.1f deg) Status=%s Cov=%.3f",
                            start_pose[0], start_pose[1], math.degrees(start_pose[2]), loc_est.status, loc_est.covariance_norm)
            else:
                sm.trigger_safe_stop(f"Valid robot localization or persistent pose unavailable (Status: {loc_est.status})")
                print("\n" + "=" * 76)
                print("[LOCALIZATION REJECTED]: No valid robot pose available in floor_map frame.")
                print(f"Status: {loc_est.status} | Covariance: {loc_est.covariance_norm:.3f}")
                print("Note: Provide an explicit start pose via --start-pose 'X Y YAW_DEG', or initialize via previous command.")
                print("=" * 76)
                sys.exit(1)

    # 3. Resolve Natural Language Prompt against Semantic Map
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
    if res_goal.selected_candidate_face:
        print(f"  Selected Face:       {res_goal.selected_candidate_face} (Clearance: {res_goal.verified_clearance_m} m)")

    # 4. Plan Collision-Free A* Route on Nav2 Costmap
    sm.start_planning()
    planner = GridPathPlanner.from_map_dir(map_dir, allow_unknown=args.allow_unknown)
    planned_path = planner.plan(
        start_pose,
        res_goal.approach_pose_2d,
        target_entity_id=res_goal.target_entity_id,
        allow_goal_snap=args.allow_goal_snap,
        max_endpoint_adjustment_m=args.max_endpoint_adj,
        robot_radius_m=None,
        target_center=res_goal.target_center_floor,
    )

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
    print(f"  Endpoint Adjustment: {planned_path.endpoint_adjustment_m:.3f} m (Tolerance: {args.max_endpoint_adj:.3f} m)")
    print(f"  Min Clearance:       {planned_path.min_clearance_m:.2f} m")
    print(f"  Footprint Validated: {planned_path.is_validated}")

    # Generate visual route artifact
    route_img_path = out_dir / "planned_goal_route.png"
    planner.visualize_planned_route(
        planned_path,
        route_img_path,
        target_center=res_goal.target_center_floor,
        target_bbox=res_goal.target_bbox_floor,
        robot_radius_m=args.robot_radius,
    )
    print(f"  Route Visual Saved:  {route_img_path}")

    # 5. Dry-Run / Preview Exit Gate
    if not is_motion_enabled:
        print("\n" + "=" * 76)
        print("DRY-RUN / PREVIEW COMPLETE")
        print("All safety checks and route plans passed. Motors remain UNPOWERED.")
        print("To execute physical motion on hardware, run with: --enable-motion")
        print("To run closed-loop sequential motion simulation, run with: --simulate-motion")
        print("=" * 76)
        sys.exit(0)

    # 6. Physical Motion Safety Verification Gate
    if args.enable_motion and not is_live_localization and not args.allow_open_loop_motion:
        sm.trigger_safe_stop("Physical execution blocked: Live measured localization stream is required")
        print("\n" + "=" * 76)
        print("[PHYSICAL MOTION BLOCKED]: Live sensor localization feedback is required.")
        print("Physical closed-loop navigation cannot execute from an open-loop or historical pose alone.")
        print("To actuate physical motors on the Raspberry Pi without camera feedback, pass --allow-open-loop-motion.")
        print("=" * 76)
        sys.exit(1)

    # 7. Start Command in Persistent State Manager
    cmd_record = state_mgr.start_command(
        prompt=args.goal,
        target_entity_id=res_goal.target_entity_id,
        target_approach_pose=res_goal.approach_pose_2d,
        start_pose=start_pose,
        start_pose_source=start_pose_source,
    )

    print("\n[ARMING MOTOR CONTROLLER & INITIALIZING LIVE OBSTACLE MONITOR]")
    motor = L298NMotorController()
    if not motor.arm():
        sm.trigger_safe_stop("Failed to arm L298N motor driver")
        state_mgr.record_interrupted(start_pose, reason="Failed to arm motor driver")
        sys.exit(1)

    obs_monitor = LiveObstacleMonitor()
    sm.arm_and_execute(enable_motion=True)
    controller = DifferentialDriveController(ControllerConfig(track_width_m=0.20))
    controller.reset()

    seq_id = 1
    current_pose = start_pose
    step_count = 0

    try:
        while sm.state in (NavigationState.EXECUTING, NavigationState.APPROACHING):
            step_count += 1

            # A. Live Obstacle Safety Check
            if args.simulate_obstacle_step is not None and step_count == args.simulate_obstacle_step:
                # Inject simulated obstacle depth return at specified distance ahead
                synth_depth = np.full((480, 640), 1800, dtype=np.uint16)
                obs_d_mm = int(args.simulate_obstacle_dist * 1000)
                synth_depth[180:320, 240:400] = obs_d_mm
                obs_monitor.update_depth_frame(synth_depth, time.time())
                logger.warning("[TEST INJECTION] Injected dynamic obstacle at %.2f m ahead!", args.simulate_obstacle_dist)

            # Check forward corridor safety
            obs_report = obs_monitor.evaluate_safety(
                current_pose_floor=current_pose,
                forward_velocity_mps=controller.last_linear_v,
            )

            if obs_report.status == ObstacleSafetyStatus.BLOCKED:
                # Immediate safety stop
                motor.set_differential_speeds(0.0, 0.0, seq_id=seq_id, timestamp_s=time.time())
                logger.warning("[OBSTACLE DETECTED]: %s. Halting for dynamic replanning...", obs_report.details)

                # Dynamically inject obstacle into costmap
                if obs_report.obstacle_point_floor is not None:
                    ox, oy = obs_report.obstacle_point_floor
                    planner.inject_obstacle(ox, oy, radius_m=0.20, inflation_radius_m=0.15)
                    logger.info("Injected obstacle at floor coords (%.2f, %.2f) into costmap.", ox, oy)

                # Attempt dynamic replan to original goal
                logger.info("Replanning route from current pose (%.2f, %.2f) to original goal...", current_pose[0], current_pose[1])
                replan = planner.plan(
                    current_pose,
                    res_goal.approach_pose_2d,
                    target_entity_id=res_goal.target_entity_id,
                    allow_goal_snap=args.allow_goal_snap,
                    max_endpoint_adjustment_m=args.max_endpoint_adj,
                    target_center=res_goal.target_center_floor,
                )

                if replan.success:
                    logger.info("[REPLAN SUCCESS]: Found detour around obstacle (New length: %.2f m). Resuming motion.", replan.path_length_m)
                    planned_path = replan
                    controller.reset()
                    # Reset obstacle monitor to clear latch
                    obs_monitor.last_depth_frame = None
                else:
                    logger.error("[REPLAN FAILED]: Path blocked by obstacle (%s). Staying safely stopped.", replan.rejection_reason)
                    sm.trigger_safe_stop(f"Path blocked by obstacle: {replan.rejection_reason}")
                    state_mgr.record_interrupted(current_pose, reason=f"Path blocked by dynamic obstacle: {replan.rejection_reason}")
                    print("\n" + "=" * 76)
                    print(f"[SAFETY HALT]: Path blocked by dynamic obstacle. No safe alternate route.")
                    print("=" * 76)
                    break

            # B. Controller Waypoint Tracking
            cmd = controller.compute_command(current_pose, planned_path.waypoints_floor)
            if cmd.phase == "APPROACHING" and sm.state == NavigationState.EXECUTING:
                sm.on_enter_approach(math.hypot(gx - current_pose[0], gy - current_pose[1]))

            if cmd.is_stopped:
                motor.set_differential_speeds(0.0, 0.0, seq_id=seq_id, timestamp_s=time.time())
                dist_to_goal = math.hypot(gx - current_pose[0], gy - current_pose[1])
                if dist_to_goal <= controller.cfg.goal_tolerance_pos_m + 0.10:
                    sm.on_goal_reached()
                    # Record completed arrival with actual measured pose
                    final_src = PoseSource.LOCALIZED if is_live_localization else PoseSource.COMMAND_ESTIMATED
                    state_mgr.record_completed(current_pose, source=final_src)
                    break
                else:
                    sm.trigger_safe_stop(f"Vehicle stopped early ({dist_to_goal:.2f}m from goal)")
                    state_mgr.record_interrupted(current_pose, reason=f"Stopped early at distance {dist_to_goal:.2f}m")
                    break

            # Send motor speeds to L298N
            motor.set_differential_speeds(
                cmd.pwm_left_pct * (1 if cmd.v_left_mps >= 0 else -1),
                cmd.pwm_right_pct * (1 if cmd.v_right_mps >= 0 else -1),
                seq_id=seq_id,
                timestamp_s=time.time(),
            )
            seq_id += 1

            # Update pose
            if args.simulate_motion or (args.enable_motion and not is_live_localization):
                dt = 0.05
                dx = cmd.linear_velocity_mps * math.cos(current_pose[2]) * dt
                dy = cmd.linear_velocity_mps * math.sin(current_pose[2]) * dt
                dyaw = cmd.angular_velocity_radps * dt
                current_pose = (current_pose[0] + dx, current_pose[1] + dy, current_pose[2] + dyaw)
                state_mgr.update_pose(current_pose, source=PoseSource.COMMAND_ESTIMATED)
            else:
                loc_feedback = localizer.get_current_pose(max_stale_duration_s=0.20)
                if not loc_feedback.is_valid or loc_feedback.status not in ("TRACKED", "DEGRADED"):
                    sm.trigger_safe_stop(f"Lost live localization feedback ({loc_feedback.status})")
                    motor.trigger_emergency_stop("Lost live localization feedback")
                    state_mgr.record_interrupted(current_pose, reason=f"Lost live localization feedback ({loc_feedback.status})")
                    print(f"\n[SAFETY HALT]: Lost live localization feedback ({loc_feedback.status}). Motors stopped.")
                    break
                current_pose = loc_feedback.robot_pose_2d
                state_mgr.update_pose(current_pose, source=PoseSource.LOCALIZED)

            # Cross-track error check
            dist_to_route = _point_to_polyline_distance(current_pose[0], current_pose[1], planned_path.waypoints_floor)
            if dist_to_route > 0.40:
                sm.trigger_safe_stop(f"Excessive path cross-track error ({dist_to_route:.2f} m > 0.40 m)")
                motor.trigger_emergency_stop("Path cross-track deviation exceeded safety threshold")
                state_mgr.record_interrupted(current_pose, reason=f"Excessive cross-track error ({dist_to_route:.2f}m)")
                print(f"\n[SAFETY HALT]: Cross-track error {dist_to_route:.2f} m exceeded limit. Emergency stop triggered.")
                break

            if step_count % 10 == 0:
                print(f"  Step {step_count}: Pose=({current_pose[0]:.2f}, {current_pose[1]:.2f}) | V={cmd.linear_velocity_mps:.2f} m/s | Phase={cmd.phase}")
            time.sleep(0.05)

        if sm.state == NavigationState.STOPPED:
            print("\n" + "=" * 76)
            print("NAVIGATION COMPLETED SUCCESSFULLY!")
            print(f"Final Robot Pose: (X={current_pose[0]:.2f} m, Y={current_pose[1]:.2f} m, Yaw={math.degrees(current_pose[2]):.1f} deg)")
            print(f"State saved to:   {state_mgr.state_file}")
            print("=" * 76)

    except KeyboardInterrupt:
        logger.warning("Manual operator interruption! Halting motors.")
        motor.trigger_emergency_stop("Manual KeyboardInterrupt")
        sm.trigger_safe_stop("Manual KeyboardInterrupt")
        state_mgr.record_interrupted(current_pose, reason="Manual operator KeyboardInterrupt")
    finally:
        motor.shutdown()


if __name__ == "__main__":
    main()
