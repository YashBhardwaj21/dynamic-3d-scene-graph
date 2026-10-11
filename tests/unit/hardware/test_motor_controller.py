"""Unit tests for L298N Motor Controller, Safety Watchdog, and Arming States."""

import time
import pytest

from scene_graph.hardware.motor_controller import L298NMotorController, L298NPinConfig


@pytest.fixture
def motor():
    cfg = L298NPinConfig(interface_type="mock", watchdog_timeout_s=0.15)
    ctrl = L298NMotorController(cfg)
    yield ctrl
    ctrl.shutdown()


def test_disarmed_rejection(motor):
    """Commands must be rejected when motor is disarmed."""
    assert motor.is_armed is False
    res = motor.set_differential_speeds(50.0, 50.0, seq_id=1)
    assert res["status"] == "REJECTED_DISARMED"
    assert motor.current_pwm_left == 0.0
    assert motor.current_pwm_right == 0.0


def test_armed_motion(motor):
    """Commands must succeed when motor is armed."""
    assert motor.arm() is True
    res = motor.set_differential_speeds(60.0, -40.0, seq_id=1)
    assert res["status"] == "APPLIED"
    assert motor.current_pwm_left == 60.0
    assert motor.current_pwm_right == 40.0
    assert motor.current_dir_left == 1
    assert motor.current_dir_right == -1


def test_emergency_stop_latch(motor):
    """E-Stop must halt motors immediately and reject commands until cleared."""
    motor.arm()
    motor.set_differential_speeds(50.0, 50.0, seq_id=1)
    assert motor.current_pwm_left > 0.0

    motor.trigger_emergency_stop("Obstacle Collision Imminent")
    assert motor.is_emergency_stopped is True
    assert motor.is_armed is False
    assert motor.current_pwm_left == 0.0
    assert motor.current_pwm_right == 0.0

    # New commands must be rejected
    res = motor.set_differential_speeds(50.0, 50.0, seq_id=2)
    assert res["status"] == "REJECTED_ESTOP"

    # Clearing E-Stop leaves robot disarmed until explicit arm
    motor.clear_emergency_stop()
    assert motor.is_emergency_stopped is False
    assert motor.is_armed is False

    res2 = motor.set_differential_speeds(50.0, 50.0, seq_id=3)
    assert res2["status"] == "REJECTED_DISARMED"


def test_stale_and_out_of_order_rejection(motor):
    """Out of order sequence IDs and stale timestamps must be rejected."""
    motor.arm()
    res1 = motor.set_differential_speeds(50.0, 50.0, seq_id=10)
    assert res1["status"] == "APPLIED"

    # Out of order seq_id (5 <= 10)
    res2 = motor.set_differential_speeds(50.0, 50.0, seq_id=5)
    assert res2["status"] == "REJECTED_OUT_OF_ORDER"

    # Stale command from 500 ms ago
    old_time = time.time() - 0.5
    res3 = motor.set_differential_speeds(50.0, 50.0, seq_id=11, timestamp_s=old_time)
    assert res3["status"] == "REJECTED_STALE"


def test_watchdog_timeout(motor):
    """Watchdog must stop motors if no new command is received within timeout."""
    motor.arm()
    motor.set_differential_speeds(50.0, 50.0, seq_id=1)
    assert motor.current_pwm_left == 50.0

    # Sleep longer than watchdog timeout (150 ms)
    time.sleep(0.22)

    # Watchdog should have auto-halted motors
    assert motor.current_pwm_left == 0.0
    assert motor.current_pwm_right == 0.0
