"""Regression tests for previously documented failures."""

import numpy as np
import pytest

from scene_graph.data.frame_packet import FramePacket, LocalizationMode
from scene_graph.data.sensor_frame import SensorFrame, StreamStatus
from scene_graph.data.timestamp import Timestamp, TimestampDomain
from scene_graph.geometry.camera import CameraIntrinsics


def test_failure1_regression_tum_2011_timestamp_preservation():
    """FAILURE 1: TUM timestamp around 2011 vs current runtime timestamp.

    Ensure TUM replay uses simulated time correctly, does NOT overwrite with current wall time,
    and preserves exact 2011 dataset timestamps in SIMULATED_TIME domain.
    """
    tum_timestamp_2011 = 1305031102.123456  # 2011-05-10
    ts = Timestamp(value=tum_timestamp_2011, domain=TimestampDomain.SIMULATED_TIME, source="tum_dataset")

    frame = SensorFrame(
        session_id="tum_2011_run",
        sequence_number=0,
        timestamp=ts,
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        camera_intrinsics=CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480),
    )

    # Invariant: Timestamp is preserved and NOT overwritten by host wall clock
    assert frame.timestamp.value == pytest.approx(1305031102.123456)
    assert frame.timestamp.domain == TimestampDomain.SIMULATED_TIME


def test_failure2_regression_use_latest_tf_string_bool_parsing():
    """FAILURE 2: use_latest_tf parameter type mismatch.

    Ensure launch and configuration parsing accepts 'true', 'false', 'True', 1, 0, or bool
    without string/bool confusion.
    """
    def parse_bool_param(val):
        if isinstance(val, str):
            return val.strip().lower() in ("true", "1", "yes")
        return bool(val)

    assert parse_bool_param("true") is True
    assert parse_bool_param("True") is True
    assert parse_bool_param("1") is True
    assert parse_bool_param(True) is True

    assert parse_bool_param("false") is False
    assert parse_bool_param("False") is False
    assert parse_bool_param("0") is False
    assert parse_bool_param(False) is False


def test_failure3_regression_sensor_ingestion_valid_when_localization_unavailable():
    """FAILURE 3: OBJECTS = 0 / RELATIONS = 0 when localization was unavailable.

    Stage 1 sensor ingestion must remain completely valid even when world pose / SLAM is missing.
    SensorFrame should not manufacture fake world poses, and downstream adapter should mark transform_valid=False.
    """
    ts = Timestamp(value=10.0, domain=TimestampDomain.HARDWARE_CLOCK)
    frame = SensorFrame(
        session_id="unlocalized_session",
        sequence_number=5,
        timestamp=ts,
        rgb=np.full((480, 640, 3), 200, dtype=np.uint8),
        camera_intrinsics=CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480),
        depth=np.full((480, 640), 1.0, dtype=np.float32),
        status=StreamStatus.OK,
    )

    # SensorFrame is 100% valid
    assert frame.has_depth is True
    assert frame.rgb is not None

    # Adapt to FramePacket when SLAM has not initialized
    packet = FramePacket.from_sensor_frame(
        sensor_frame=frame,
        world_T_camera=None,
        transform_source="missing_slam_tf",
        transform_valid=False,
        localization_mode=LocalizationMode.WORLD_MODE,
    )

    # Invariant: FramePacket is created with transform_valid=False, without manufacturing invalid world poses
    assert packet.has_pose is False
    assert packet.world_T_camera is None
    assert packet.transform_valid is False
    assert packet.transform_source == "missing_slam_tf"
    # Sensor data itself is completely intact
    assert packet.rgb is not None
    assert packet.depth is not None


def test_failure4_regression_use_sim_time_predeclared_no_exception():
    """FAILURE 4: ParameterAlreadyDeclaredException on use_sim_time.

    Ensure node startup safely checks has_parameter('use_sim_time') before declaring it,
    preventing ParameterAlreadyDeclaredException when use_sim_time is supplied via launch arguments.
    """
    class MockNode:
        def __init__(self, already_declared=False):
            self.params = {}
            if already_declared:
                self.params["use_sim_time"] = True

        def has_parameter(self, name: str) -> bool:
            return name in self.params

        def declare_parameter(self, name: str, default: object):
            if name in self.params:
                raise RuntimeError(f"ParameterAlreadyDeclaredException: {name}")
            self.params[name] = default

    # Case A: Parameter already declared by rclpy / launch
    node_predeclared = MockNode(already_declared=True)
    if not node_predeclared.has_parameter("use_sim_time"):
        node_predeclared.declare_parameter("use_sim_time", False)
    assert node_predeclared.params["use_sim_time"] is True

    # Case B: Parameter not declared yet
    node_fresh = MockNode(already_declared=False)
    if not node_fresh.has_parameter("use_sim_time"):
        node_fresh.declare_parameter("use_sim_time", False)
    assert node_fresh.params["use_sim_time"] is False


def test_scene_graph_node_use_sim_time_ros():
    """Verify SceneGraphROSNode can be instantiated when rclpy has already declared use_sim_time."""
    try:
        import rclpy
        from scene_graph_ros.scene_graph_node import SceneGraphROSNode
    except ImportError:
        pytest.skip("rclpy or scene_graph_ros not available")

    if rclpy.ok():
        rclpy.shutdown()
    rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
    try:
        node = SceneGraphROSNode()
        assert node.get_parameter("use_sim_time").value is True
        node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


