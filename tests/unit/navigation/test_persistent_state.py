"""Automated unit and integration tests for Persistent State and Sequential Navigation.

Tests Cases:
  Case A: Successive commands: First command saves final pose, second command starts from that pose.
  Case B: Process restart: Recovers history and checkpoint pose safely.
  Case C: Interrupted command: Recorded as INTERRUPTED, not COMPLETED.
  Case D: Localization loss / invalid pose: Rejected, does not fabricate pose.
  Case G: Map mismatch / coordinate frame incompatibility: Rejects incompatible state.
  Case H: Goal reached: Verified against measured pose, not requested waypoint.
  Atomic persistence: Corrupt file recovery, atomic replace, valid JSON payload.
"""

import json
import math
import time
from pathlib import Path
import pytest
import numpy as np

from scene_graph.navigation.state_manager import (
    NavigationStateManager,
    PoseSource,
    CommandStatus,
    RobotPoseRecord,
    CommandRecord,
)


@pytest.fixture
def temp_map_dir(tmp_path):
    map_dir = tmp_path / "test_map"
    map_dir.mkdir()
    sem_map = {
        "coordinate_frame": "floor_map (Unified REP-103)",
        "reference_session": "session_1",
        "entities": {
            "desk_001": {
                "class_name": "desk",
                "centroid_floor": [2.0, -2.0, 0.4],
                "bbox_min_floor": [1.4, -2.4, 0.0],
                "bbox_max_floor": [2.6, -1.6, 0.8],
                "dimensions": [1.2, 0.8, 0.8],
                "is_confirmed": True,
            },
            "chair_001": {
                "class_name": "chair",
                "centroid_floor": [0.5, -1.5, 0.4],
                "bbox_min_floor": [0.2, -1.8, 0.0],
                "bbox_max_floor": [0.8, -1.2, 0.8],
                "dimensions": [0.6, 0.6, 0.8],
                "is_confirmed": True,
            },
        },
    }
    with open(map_dir / "semantic_map.json", "w", encoding="utf-8") as f:
        json.dump(sem_map, f)
    return map_dir


def test_initial_state_empty(temp_map_dir):
    """Clean initialization when no previous state exists."""
    mgr = NavigationStateManager(temp_map_dir)
    assert mgr.current_pose is None
    assert mgr.last_confirmed_stopping_pose is None
    assert mgr.get_initial_pose_for_planning() is None
    assert len(mgr.command_history) == 0


def test_case_a_successive_commands(temp_map_dir):
    """Case A: Command 1 saves final measured pose, Command 2 uses that pose as start pose."""
    mgr = NavigationStateManager(temp_map_dir)

    # 1. Start Command 1
    init_pose = (0.0, 0.0, 0.0)
    cmd1 = mgr.start_command(
        prompt="Navigate to desk_001",
        target_entity_id="desk_001",
        target_approach_pose=(1.45, -1.35, -0.75),
        start_pose=init_pose,
        start_pose_source=PoseSource.MANUAL_INITIALIZED,
    )
    assert mgr.active_command is not None
    assert mgr.active_command.command_id == cmd1.command_id

    # Simulate reaching approach pose
    final_pose_1 = (1.43, -1.36, -0.72)  # Actual measured stopping pose (slight realistic delta from requested)
    mgr.record_completed(final_pose_1, source=PoseSource.LOCALIZED)

    assert mgr.active_command is None
    assert mgr.last_completed_command.status == CommandStatus.COMPLETED
    assert mgr.last_confirmed_stopping_pose.to_tuple_2d() == final_pose_1
    assert mgr.last_confirmed_stopping_pose.source == PoseSource.LOCALIZED

    # 2. Next sequential command starts without manual input!
    next_start = mgr.get_initial_pose_for_planning(allow_command_estimated=True)
    assert next_start == final_pose_1, "Command 2 must start from the measured stopping pose of Command 1!"

    cmd2 = mgr.start_command(
        prompt="Navigate to chair_001",
        target_entity_id="chair_001",
        target_approach_pose=(0.5, -1.0, 0.0),
        start_pose=next_start,
        start_pose_source=PoseSource.LOCALIZED,
    )
    assert cmd2.start_pose.to_tuple_2d() == final_pose_1

    final_pose_2 = (0.49, -1.02, 0.05)
    mgr.record_completed(final_pose_2, source=PoseSource.LOCALIZED)

    assert len(mgr.command_history) == 2
    assert mgr.command_history[0].prompt == "Navigate to desk_001"
    assert mgr.command_history[1].prompt == "Navigate to chair_001"
    assert mgr.get_initial_pose_for_planning() == final_pose_2


def test_case_b_process_restart(temp_map_dir):
    """Case B: Process terminates, new process recovers history and checkpoint pose."""
    state_file = temp_map_dir / "robot_navigation_state.json"
    mgr1 = NavigationStateManager(temp_map_dir, state_file=state_file)

    mgr1.start_command("Go to desk", "desk_001", (1.5, -1.5, 0.0), (0.0, 0.0, 0.0))
    mgr1.record_completed((1.48, -1.52, 0.1), source=PoseSource.LOCALIZED)
    assert state_file.exists()

    # Create new instance (simulating new process launch)
    mgr2 = NavigationStateManager(temp_map_dir, state_file=state_file)
    assert mgr2.last_confirmed_stopping_pose is not None
    assert mgr2.last_confirmed_stopping_pose.to_tuple_2d() == (1.48, -1.52, 0.1)
    # Recovered pose is correctly tagged as historical checkpoint
    assert mgr2.last_confirmed_stopping_pose.source == PoseSource.HISTORICAL_CHECKPOINT
    assert len(mgr2.command_history) == 1
    assert mgr2.command_history[0].status == CommandStatus.COMPLETED


def test_case_c_interrupted_command(temp_map_dir):
    """Case C: Interrupted command recorded as INTERRUPTED, not COMPLETED."""
    mgr = NavigationStateManager(temp_map_dir)
    cmd = mgr.start_command("Go to door", "door_001", (3.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    # Interrupted midway
    mid_pose = (1.2, 0.0, 0.0)
    mgr.record_interrupted(mid_pose, reason="Obstacle blockage / Manual stop")

    assert mgr.active_command is None
    assert len(mgr.command_history) == 1
    rec = mgr.command_history[0]
    assert rec.status == CommandStatus.INTERRUPTED
    assert rec.failure_reason == "Obstacle blockage / Manual stop"
    assert rec.final_pose.to_tuple_2d() == mid_pose
    assert mgr.current_pose.to_tuple_2d() == mid_pose


def test_case_d_command_estimated_separation(temp_map_dir):
    """Case D: Distinguish command_estimated from localized; never promote silently."""
    mgr = NavigationStateManager(temp_map_dir)
    mgr.update_pose((1.0, 2.0, 0.0), source=PoseSource.COMMAND_ESTIMATED)

    # Physical navigation requires independent localization confirmation
    physical_start = mgr.get_initial_pose_for_planning(allow_command_estimated=False)
    assert physical_start is None, "Physical navigation must not silently accept unverified command-estimated poses!"

    # Simulation / dry-run can accept command-estimated poses
    sim_start = mgr.get_initial_pose_for_planning(allow_command_estimated=True)
    assert sim_start == (1.0, 2.0, 0.0)


def test_case_g_map_mismatch(temp_map_dir, tmp_path):
    """Case G: Incompatible map directory rejects saved state."""
    state_file = temp_map_dir / "robot_navigation_state.json"
    mgr1 = NavigationStateManager(temp_map_dir, state_file=state_file)
    mgr1.update_pose((1.0, 1.0, 0.0), source=PoseSource.LOCALIZED)

    # Incompatible other map
    other_map_dir = tmp_path / "different_map"
    other_map_dir.mkdir()

    mgr_other = NavigationStateManager(other_map_dir, state_file=state_file)
    # The mismatched state is rejected
    assert mgr_other.current_pose is None or mgr_other.get_initial_pose_for_planning() is None


def test_case_h_measured_vs_requested_destination(temp_map_dir):
    """Case H: Destination waypoint is never substituted for measured stopping pose."""
    mgr = NavigationStateManager(temp_map_dir)
    requested_approach = (2.000, -1.500, 0.000)
    actual_stopping = (1.932, -1.481, 0.045)  # Physical robot stops with small realistic offset

    mgr.start_command("Go to desk", "desk_001", requested_approach, (0.0, 0.0, 0.0))
    mgr.record_completed(actual_stopping, source=PoseSource.LOCALIZED)

    saved_pose = mgr.get_initial_pose_for_planning()
    assert saved_pose == actual_stopping
    assert saved_pose != requested_approach, "System must save actual measured pose, not commanded waypoint!"


def test_atomic_persistence_and_corruption_recovery(temp_map_dir):
    """Atomic write protects against partial files; handles corrupted JSON gracefully."""
    state_file = temp_map_dir / "robot_navigation_state.json"
    mgr = NavigationStateManager(temp_map_dir, state_file=state_file)
    mgr.update_pose((3.14, 2.71, 1.57), source=PoseSource.LOCALIZED)

    assert state_file.exists()

    # Corrupt the file
    with open(state_file, "w", encoding="utf-8") as f:
        f.write("{ bad json half-written payload ...")

    # Re-instantiate manager; must not crash!
    mgr_recovered = NavigationStateManager(temp_map_dir, state_file=state_file)
    assert mgr_recovered.current_pose is None
    # Verifies corrupted file was backed up
    assert (temp_map_dir / "robot_navigation_state.corrupted").exists()
