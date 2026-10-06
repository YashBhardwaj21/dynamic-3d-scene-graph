"""Comprehensive Stage 3 test suite for pose estimation and estimator subsystems.

Covers:
A. PoseEstimate contract and invariants (SE(3), covariance, timestamps, invalid semantics)
B. EstimatorState contract and tracking states (inlier ratio, defaults, telemetry age)
C. TFBufferEstimator (exact lookup, causal fallback, future TF rejection, stale TF rejection, missing TF)
D. RTABMapEstimator telemetry (OdomInfo, Info, covariance, tracking loss overrides frozen TF, recovery, loop closure)
E. FramePacket integration and two-way legacy field synchronization
F. IdentityEstimator (camera-local mode)
G. GroundTruthEstimator (TUM trajectory matching, timestamp skew rejection)
"""

from __future__ import annotations

import numpy as np
import pytest

from scene_graph.data.frame_packet import FramePacket, LocalizationMode
from scene_graph.data.pose_estimate import EstimatorState, PoseEstimate, TrackingState
from scene_graph.data.sensor_frame import SensorFrame, StreamStatus
from scene_graph.data.timestamp import Timestamp, TimestampDomain
from scene_graph.estimation.ground_truth import GroundTruthEstimator
from scene_graph.estimation.identity import IdentityEstimator
from scene_graph.estimation.rtabmap import RTABMapEstimator
from scene_graph.estimation.tf_buffer import TFBufferEstimator
from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.transforms import quaternion_to_matrix


# Helper Fixtures and Test Doubles

class MockTFTransform:
    def __init__(self, x=0.0, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        class Translation:
            pass
        class Rotation:
            pass
        self.translation = Translation()
        self.translation.x = x
        self.translation.y = y
        self.translation.z = z
        self.rotation = Rotation()
        self.rotation.x = qx
        self.rotation.y = qy
        self.rotation.z = qz
        self.rotation.w = qw


class MockTransformStamped:
    def __init__(self, sec: int, nanosec: int, frame_id="world", child_frame_id="camera_optical_frame",
                 tx=1.0, ty=2.0, tz=3.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        class Stamp:
            def __init__(self, s, ns):
                self.sec = s
                self.nanosec = ns
        class Header:
            def __init__(self, s, ns, fid):
                self.stamp = Stamp(s, ns)
                self.frame_id = fid

        self.header = Header(sec, nanosec, frame_id)
        self.child_frame_id = child_frame_id
        self.transform = MockTFTransform(tx, ty, tz, qx, qy, qz, qw)


class MockTFBuffer:
    """Mock tf2_ros.Buffer for testing exact, causal, future, and stale TF lookups."""

    def __init__(self):
        self.transforms: list[tuple[float, MockTransformStamped]] = []

    def add_transform(self, timestamp: float, tx=1.0, ty=2.0, tz=3.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        sec = int(timestamp)
        nanosec = int(round((timestamp - sec) * 1e9))
        stamped = MockTransformStamped(sec, nanosec, tx=tx, ty=ty, tz=tz, qx=qx, qy=qy, qz=qz, qw=qw)
        self.transforms.append((timestamp, stamped))
        self.transforms.sort(key=lambda item: item[0])

    def lookup_transform(self, target_frame: str, source_frame: str, time_val: any, timeout: any = None):
        if not self.transforms:
            raise RuntimeError("TF lookup failed: buffer empty")

        # Convert time_val to float timestamp
        if hasattr(time_val, "sec") and hasattr(time_val, "nanosec"):
            t_sec = float(time_val.sec) + float(time_val.nanosec) * 1e-9
        elif isinstance(time_val, (int, float)):
            t_sec = float(time_val)
        else:
            # Time() with 0 or empty means latest
            t_sec = 0.0

        if t_sec == 0.0:
            # Return newest transform in buffer
            return self.transforms[-1][1]

        # Exact match within 1ms
        for ts, stamped in self.transforms:
            if abs(ts - t_sec) <= 1e-3:
                return stamped

        raise RuntimeError(f"TF lookup failed: no transform at {t_sec:.4f}")


def create_dummy_sensor_frame(timestamp: float = 100.0, frame_id: str = "camera_optical_frame") -> SensorFrame:
    return SensorFrame(
        session_id="test_session",
        sequence_number=1,
        timestamp=Timestamp(value=timestamp, domain=TimestampDomain.SIMULATED_TIME, source="test"),
        rgb=np.zeros((60, 80, 3), dtype=np.uint8),
        depth=np.ones((60, 80), dtype=np.float32) * 1.5,
        camera_intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=40.0, cy=30.0, width=80, height=60),
        frame_id=frame_id,
        optical_frame_id=frame_id,
        status=StreamStatus.OK,
    )


# Section A: PoseEstimate Invariants and Validations

class TestPoseEstimateContract:
    def test_valid_se3_construction(self):
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = [0.5, -0.2, 1.3]
        pose = PoseEstimate(
            world_T_camera=T,
            timestamp=10.0,
            valid=True,
            age=0.015,
            source="rtabmap",
            frame_id="world",
            pose_timestamp=9.985,
            target_timestamp=10.0,
            lookup_mode="exact",
        )
        assert pose.valid is True
        assert np.allclose(pose.translation, [0.5, -0.2, 1.3])
        assert np.allclose(pose.rotation, np.eye(3))
        assert pose.age == pytest.approx(0.015)
        assert pose.is_finite is True

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            PoseEstimate(world_T_camera=np.eye(3), timestamp=10.0, valid=True)

    def test_non_finite_raises(self):
        T = np.eye(4)
        T[0, 3] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            PoseEstimate(world_T_camera=T, timestamp=10.0, valid=True)

    def test_invalid_bottom_row_raises(self):
        T = np.eye(4)
        T[3, 0] = 0.5
        with pytest.raises(ValueError, match="bottom row"):
            PoseEstimate(world_T_camera=T, timestamp=10.0, valid=True)

    def test_invalid_rotation_raises(self):
        T = np.eye(4)
        T[:3, :3] = np.eye(3) * 2.0  # det = 8, not rotation
        with pytest.raises(ValueError, match="orthonormal|determinant"):
            PoseEstimate(world_T_camera=T, timestamp=10.0, valid=True)

    def test_covariance_shape_validation(self):
        T = np.eye(4)
        # Valid 6x6
        pose = PoseEstimate(world_T_camera=T, timestamp=10.0, valid=True, covariance=np.eye(6))
        assert pose.covariance.shape == (6, 6)

        # Invalid shape raises
        with pytest.raises(ValueError, match="covariance"):
            PoseEstimate(world_T_camera=T, timestamp=10.0, valid=True, covariance=np.eye(5))

    def test_invalid_constructor_helper(self):
        pose = PoseEstimate.invalid(
            timestamp=10.0,
            source="rtabmap",
            target_timestamp=10.0,
            transform_source="stale_slam_tf",
            frame_id="world",
        )
        assert pose.valid is False
        assert pose.world_T_camera is None
        assert pose.source == "rtabmap"
        assert pose.transform_source == "stale_slam_tf"


# Section B: EstimatorState Invariants and Tracking Enums

class TestEstimatorStateContract:
    def test_tracking_states_enum(self):
        assert TrackingState.INITIALIZING.name == "INITIALIZING"
        assert TrackingState.TRACKING.name == "TRACKING"
        assert TrackingState.LOST.name == "LOST"
        assert TrackingState.RECOVERED.name == "RECOVERED"
        assert TrackingState.TRACKING.value == "tracking"

    def test_inlier_ratio_and_defaults(self):
        state = EstimatorState(
            backend_name="rtabmap",
            tracking_state=TrackingState.TRACKING,
            matches=500,
            inliers=400,
            features=800,
            local_map_size=120,
            odometry_lost=False,
        )
        assert state.inliers == 400
        assert state.features == 800
        assert state.matches == 500
        assert state.is_healthy is True

    def test_initializing_and_lost_helpers(self):
        s_init = EstimatorState.initializing("rtabmap")
        assert s_init.tracking_state == TrackingState.INITIALIZING
        assert s_init.is_healthy is False

        s_lost = EstimatorState.lost("rtabmap", odometry_lost=True)
        assert s_lost.tracking_state == TrackingState.LOST
        assert s_lost.odometry_lost is True
        assert s_lost.is_healthy is False


# Section C: TFBufferEstimator Policy Tests

class TestTFBufferEstimator:
    def test_exact_pose_lookup(self):
        buf = MockTFBuffer()
        buf.add_transform(timestamp=10.0, tx=1.0, ty=2.0, tz=3.0)
        estimator = TFBufferEstimator(tf_buffer=buf, world_frame="world", sensor_frame="camera_optical_frame", pose_max_dt=0.05)

        sf = create_dummy_sensor_frame(timestamp=10.0)
        pose, state = estimator.estimate(sf)

        assert pose.valid is True
        assert pose.lookup_mode == "exact"
        assert pose.source == "tf_buffer"
        assert pose.age == pytest.approx(0.0)
        assert np.allclose(pose.translation, [1.0, 2.0, 3.0])
        assert state.tracking_state == TrackingState.TRACKING

    def test_causal_latest_fallback(self):
        buf = MockTFBuffer()
        # Transform at t=9.95 (50ms before frame at t=10.0)
        buf.add_transform(timestamp=9.95, tx=0.5, ty=1.0, tz=1.5)
        estimator = TFBufferEstimator(
            tf_buffer=buf,
            world_frame="world",
            sensor_frame="camera_optical_frame",
            pose_max_dt=0.02,  # exact fails because 50ms > 20ms
            slam_pose_max_age=0.20,  # causal latest accepts (50ms <= 200ms)
            allow_causal_fallback=True,
        )

        sf = create_dummy_sensor_frame(timestamp=10.0)
        pose, state = estimator.estimate(sf)

        assert pose.valid is True
        assert pose.lookup_mode == "causal_latest"
        assert pose.age == pytest.approx(0.05, abs=1e-3)
        assert state.tracking_state == TrackingState.TRACKING

    def test_future_transform_rejected(self):
        buf = MockTFBuffer()
        # Transform at t=10.2 (future relative to frame at t=10.0)
        buf.add_transform(timestamp=10.2, tx=1.0, ty=1.0, tz=1.0)
        estimator = TFBufferEstimator(
            tf_buffer=buf,
            world_frame="world",
            sensor_frame="camera_optical_frame",
            pose_max_dt=0.05,
            allow_causal_fallback=True,
        )

        sf = create_dummy_sensor_frame(timestamp=10.0)
        pose, state = estimator.estimate(sf)

        assert pose.valid is False
        assert pose.transform_source == "future_tf_rejected"
        assert state.tracking_state == TrackingState.INITIALIZING

    def test_stale_transform_rejected(self):
        buf = MockTFBuffer()
        # Transform at t=9.0 (1.0s before frame at t=10.0, exceeds slam_pose_max_age=0.20s)
        buf.add_transform(timestamp=9.0, tx=1.0, ty=1.0, tz=1.0)
        estimator = TFBufferEstimator(
            tf_buffer=buf,
            world_frame="world",
            sensor_frame="camera_optical_frame",
            pose_max_dt=0.05,
            slam_pose_max_age=0.20,
            allow_causal_fallback=True,
        )

        sf = create_dummy_sensor_frame(timestamp=10.0)
        pose, state = estimator.estimate(sf)

        assert pose.valid is False
        assert pose.transform_source == "stale_slam_tf"
        assert state.tracking_state == TrackingState.INITIALIZING

    def test_missing_tf_returns_invalid(self):
        buf = MockTFBuffer()  # Empty
        estimator = TFBufferEstimator(tf_buffer=buf, world_frame="world", sensor_frame="camera_optical_frame")

        sf = create_dummy_sensor_frame(timestamp=10.0)
        pose, state = estimator.estimate(sf)

        assert pose.valid is False
        assert state.tracking_state == TrackingState.INITIALIZING


# Section D: RTABMapEstimator Telemetry, Covariance, and State Machine

class TestRTABMapEstimator:
    def test_odom_info_ingestion_and_covariance(self):
        buf = MockTFBuffer()
        buf.add_transform(timestamp=10.0, tx=0.1, ty=0.2, tz=0.3)
        estimator = RTABMapEstimator(
            tf_buffer=buf,
            world_frame="world",
            sensor_frame="camera_optical_frame",
            pose_max_dt=0.05,
            slam_pose_max_age=0.20,
            telemetry_max_age=0.25,
        )

        # Ingest causal telemetry at t=9.98
        cov_flat = list(np.eye(6).flatten() * 0.01)
        estimator.add_odom_info_sample(
            timestamp=9.98,
            lost=False,
            matches=350,
            inliers=280,
            features=500,
            local_map_size=85,
            local_key_frames=12,
            key_frame_added=True,
            time_estimation_sec=0.018,
            covariance=cov_flat,
        )

        sf = create_dummy_sensor_frame(timestamp=10.0)
        pose, state = estimator.estimate(sf)

        assert pose.valid is True
        assert pose.covariance is not None
        assert pose.covariance.shape == (6, 6)
        assert np.isclose(pose.covariance[0, 0], 0.01)

        assert state.backend_name == "rtabmap"
        assert state.tracking_state == TrackingState.TRACKING
        assert state.inliers == 280
        assert state.features == 500
        assert state.matches == 350
        assert state.inlier_ratio == pytest.approx(280.0 / 500.0)
        assert state.local_map_size == 85
        assert state.key_frame_added is True
        assert state.registration_time_ms == pytest.approx(18.0)
        assert state.telemetry_age == pytest.approx(0.02, abs=1e-3)
        assert state.covariance_available is True

    def test_odometry_lost_overrides_frozen_tf(self):
        """Hard requirement: If RTAB-Map reports odometry lost, do NOT treat frozen TF as healthy."""
        buf = MockTFBuffer()
        # TF exists and is temporally recent
        buf.add_transform(timestamp=9.95, tx=1.0, ty=2.0, tz=3.0)
        estimator = RTABMapEstimator(
            tf_buffer=buf,
            world_frame="world",
            sensor_frame="camera_optical_frame",
            pose_max_dt=0.05,
            slam_pose_max_age=0.20,
            telemetry_max_age=0.25,
        )

        # First frame: healthy
        estimator.add_odom_info_sample(timestamp=9.95, lost=False, inliers=200, features=300)
        sf1 = create_dummy_sensor_frame(timestamp=9.95)
        pose1, state1 = estimator.estimate(sf1)
        assert pose1.valid is True
        assert state1.tracking_state == TrackingState.TRACKING

        # Second frame: TF is still present in buffer at t=9.95, but RTAB-Map reports lost at t=10.0!
        estimator.add_odom_info_sample(timestamp=10.0, lost=True, inliers=0, features=50)
        sf2 = create_dummy_sensor_frame(timestamp=10.0)
        pose2, state2 = estimator.estimate(sf2)

        # Invariant check:
        assert pose2.valid is False, "Frozen TF must be invalidated when odometry is lost"
        assert pose2.transform_source == "rtabmap_odom_lost"
        assert state2.tracking_state == TrackingState.LOST
        assert state2.odometry_lost is True

    def test_tracking_recovery_transition(self):
        """Verify: LOST -> RECOVERED (1 frame) -> TRACKING."""
        buf = MockTFBuffer()
        estimator = RTABMapEstimator(
            tf_buffer=buf,
            world_frame="world",
            sensor_frame="camera_optical_frame",
            slam_pose_max_age=0.20,
            telemetry_max_age=0.25,
        )

        # 1. Healthy frame
        buf.add_transform(timestamp=1.0)
        estimator.add_odom_info_sample(timestamp=1.0, lost=False, inliers=100)
        p1, s1 = estimator.estimate(create_dummy_sensor_frame(1.0))
        assert s1.tracking_state == TrackingState.TRACKING

        # 2. Lost frame
        buf.add_transform(timestamp=2.0)
        estimator.add_odom_info_sample(timestamp=2.0, lost=True)
        p2, s2 = estimator.estimate(create_dummy_sensor_frame(2.0))
        assert s2.tracking_state == TrackingState.LOST

        # 3. Recovery frame (first frame with valid pose + healthy odom after LOST)
        buf.add_transform(timestamp=3.0)
        estimator.add_odom_info_sample(timestamp=3.0, lost=False, inliers=150)
        p3, s3 = estimator.estimate(create_dummy_sensor_frame(3.0))
        assert s3.tracking_state == TrackingState.RECOVERED
        assert s3.tracking_recovered is True

        # 4. Normal tracking frame after recovery
        buf.add_transform(timestamp=4.0)
        estimator.add_odom_info_sample(timestamp=4.0, lost=False, inliers=160)
        p4, s4 = estimator.estimate(create_dummy_sensor_frame(4.0))
        assert s4.tracking_state == TrackingState.TRACKING
        assert s4.tracking_recovered is False

    def test_loop_closure_detection_flag(self):
        buf = MockTFBuffer()
        buf.add_transform(timestamp=10.0)
        estimator = RTABMapEstimator(tf_buffer=buf, world_frame="world", sensor_frame="camera_optical_frame")

        estimator.add_info_sample(timestamp=10.0, ref_id=42, loop_closure_id=105)
        estimator.add_odom_info_sample(timestamp=10.0, lost=False, inliers=100)

        p, s = estimator.estimate(create_dummy_sensor_frame(10.0))
        assert s.loop_closure_detected is True
        assert s.loop_closure_id == 105
        assert s.current_node_id == 42
        assert s.tracking_state == TrackingState.TRACKING

        # Next frame with same loop_closure_id should NOT re-trigger event flag
        buf.add_transform(timestamp=10.1)
        estimator.add_info_sample(timestamp=10.1, ref_id=43, loop_closure_id=105)
        estimator.add_odom_info_sample(timestamp=10.1, lost=False, inliers=100)

        p2, s2 = estimator.estimate(create_dummy_sensor_frame(10.1))
        assert s2.loop_closure_detected is False
        assert s2.loop_closure_id == 105

    def test_future_telemetry_strictly_ignored(self):
        buf = MockTFBuffer()
        buf.add_transform(timestamp=10.0)
        estimator = RTABMapEstimator(tf_buffer=buf, world_frame="world", sensor_frame="camera_optical_frame")

        # Ingest telemetry at t=10.5 (future relative to frame at t=10.0)
        estimator.add_odom_info_sample(timestamp=10.5, lost=False, inliers=999)

        p, s = estimator.estimate(create_dummy_sensor_frame(10.0))
        # Telemetry should NOT be used because 10.5 > 10.0
        assert s.inliers == 0
        assert s.telemetry_age == float("inf")


# Section E: FramePacket Synchronization and Invariants

class TestFramePacketIntegration:
    def test_frame_packet_synchronizes_pose_estimate(self):
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        pose = PoseEstimate(
            world_T_camera=T,
            timestamp=10.0,
            valid=True,
            age=0.02,
            source="rtabmap",
            frame_id="world",
            pose_timestamp=9.98,
        )
        state = EstimatorState(
            backend_name="rtabmap",
            tracking_state=TrackingState.TRACKING,
            inliers=150,
            features=300,
        )
        sf = create_dummy_sensor_frame(timestamp=10.0)

        packet = FramePacket.from_sensor_frame(
            sensor_frame=sf,
            pose_estimate=pose,
            estimator_state=state,
            world_frame="world",
        )

        # Verify legacy fields are perfectly synchronized with PoseEstimate
        assert packet.transform_valid is True
        assert np.allclose(packet.world_T_camera, T)
        assert packet.pose_timestamp == pytest.approx(9.98)
        assert packet.pose_age == pytest.approx(0.02)
        assert packet.transform_source == "rtabmap"
        assert packet.estimator_state.tracking_state == TrackingState.TRACKING

    def test_legacy_arguments_populate_pose_estimate(self):
        T = np.eye(4)
        T[:3, 3] = [0.5, 0.5, 0.5]
        sf = create_dummy_sensor_frame(timestamp=10.0)

        packet = FramePacket.from_sensor_frame(
            sensor_frame=sf,
            world_T_camera=T,
            pose_timestamp=9.99,
            transform_source="tf_exact",
            transform_valid=True,
        )

        assert packet.pose_estimate is not None
        assert packet.pose_estimate.valid is True
        assert np.allclose(packet.pose_estimate.world_T_camera, T)
        assert packet.pose_estimate.pose_timestamp == 9.99
        assert packet.pose_estimate.source == "tf_exact"


# Section F: IdentityEstimator (Camera-Local Fallback)

class TestIdentityEstimator:
    def test_camera_local_mode(self):
        estimator = IdentityEstimator(frame_id="camera_optical_frame")
        sf = create_dummy_sensor_frame(timestamp=10.0)

        pose, state = estimator.estimate(sf)

        assert pose.valid is True
        assert np.allclose(pose.world_T_camera, np.eye(4))
        assert pose.source == "camera_local"
        assert pose.frame_id == "camera_optical_frame"
        assert state.backend_name == "camera_local"
        assert state.tracking_state == TrackingState.TRACKING

        # Pass to FramePacket and verify is_persistent_world_frame is False
        packet = FramePacket.from_sensor_frame(
            sensor_frame=sf,
            pose_estimate=pose,
            estimator_state=state,
            localization_mode=LocalizationMode.CAMERA_LOCAL_MODE,
            world_frame="camera_optical_frame",
        )
        assert packet.is_persistent_world_frame is False


# Section G: GroundTruthEstimator

class TestGroundTruthEstimator:
    def test_gt_trajectory_association(self, tmp_path):
        gt_file = tmp_path / "groundtruth.txt"
        # TUM format: timestamp tx ty tz qx qy qz qw
        gt_file.write_text(
            "# TUM ground truth\n"
            "10.0000 1.0 2.0 3.0 0.0 0.0 0.0 1.0\n"
            "10.0333 1.1 2.1 3.1 0.0 0.0 0.0 1.0\n"
            "10.0666 1.2 2.2 3.2 0.0 0.0 0.0 1.0\n"
        )

        estimator = GroundTruthEstimator(trajectory_file=gt_file, max_dt=0.02)

        # Query close to 10.0333
        sf = create_dummy_sensor_frame(timestamp=10.035)
        pose, state = estimator.estimate(sf)

        assert pose.valid is True
        assert np.allclose(pose.translation, [1.1, 2.1, 3.1])
        assert pose.source == "ground_truth"
        assert state.tracking_state == TrackingState.TRACKING

    def test_gt_out_of_sync_rejection(self, tmp_path):
        gt_file = tmp_path / "groundtruth.txt"
        gt_file.write_text("10.0 1.0 2.0 3.0 0.0 0.0 0.0 1.0\n")

        estimator = GroundTruthEstimator(trajectory_file=gt_file, max_dt=0.02)

        # Query far away from 10.0 (e.g. at 10.5)
        sf = create_dummy_sensor_frame(timestamp=10.5)
        pose, state = estimator.estimate(sf)

        assert pose.valid is False
        assert state.tracking_state == TrackingState.LOST
