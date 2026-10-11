"""Unit tests for Autonomous Navigation Safety State Machine."""

import time
import pytest

from scene_graph.navigation.state_machine import NavigationStateMachine, NavigationState


def test_normal_lifecycle():
    sm = NavigationStateMachine(max_localization_covariance=0.05, max_sensor_age_s=0.25)
    assert sm.state == NavigationState.DISABLED

    assert sm.initialize() is True
    assert sm.state == NavigationState.INITIALIZING

    assert sm.on_sensors_ready() is True
    assert sm.state == NavigationState.LOCALIZING

    # Localization acquisition
    now = time.time()
    assert sm.on_localization_update(pose_valid=True, covariance=0.02, timestamp_s=now) is True
    assert sm.state == NavigationState.READY

    # Planning
    assert sm.start_planning() is True
    assert sm.state == NavigationState.PLANNING

    assert sm.on_plan_ready(path_found=True) is True
    assert sm.state == NavigationState.READY

    # Dry-run execution blocked
    assert sm.arm_and_execute(enable_motion=False) is False
    assert sm.state == NavigationState.READY

    # Real execution armed
    assert sm.arm_and_execute(enable_motion=True) is True
    assert sm.state == NavigationState.EXECUTING
    assert sm.is_motion_enabled is True

    # Entering approach zone
    assert sm.on_enter_approach(distance_to_goal_m=0.5) is True
    assert sm.state == NavigationState.APPROACHING

    # Final arrival
    assert sm.on_goal_reached() is True
    assert sm.state == NavigationState.STOPPED
    assert sm.is_motion_enabled is False


def test_lost_localization_during_execution():
    """Localization loss during motion must latch into SAFE_STOP immediately."""
    sm = NavigationStateMachine()
    sm.initialize()
    sm.on_sensors_ready()
    now = time.time()
    sm.on_localization_update(pose_valid=True, covariance=0.01, timestamp_s=now)
    sm.arm_and_execute(enable_motion=True)
    assert sm.state == NavigationState.EXECUTING

    # Simulate lost localization
    sm.on_localization_update(pose_valid=False, covariance=0.99, timestamp_s=time.time())
    assert sm.state == NavigationState.SAFE_STOP
    assert sm.is_motion_enabled is False
    assert "Localization lost" in sm.fault_reason

    # Recovery
    assert sm.recover() is True
    assert sm.state == NavigationState.STOPPED


def test_stale_sensor_trigger():
    """Stale timestamp during motion must trigger SAFE_STOP."""
    sm = NavigationStateMachine(max_sensor_age_s=0.20)
    sm.initialize()
    sm.on_sensors_ready()
    sm.on_localization_update(pose_valid=True, covariance=0.01, timestamp_s=time.time())
    sm.arm_and_execute(enable_motion=True)

    # Update with timestamp from 500ms ago
    stale_ts = time.time() - 0.50
    sm.on_localization_update(pose_valid=True, covariance=0.01, timestamp_s=stale_ts)
    assert sm.state == NavigationState.SAFE_STOP
    assert "stale" in sm.fault_reason
