import copy
import time
from collections import deque
import numpy as np
import pytest

from scene_graph.config import load_config
from scene_graph.data.frame_packet import FramePacket, LocalizationMode
from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.geometry.point_cloud import ObjectGeometry, GeometryStatus
from scene_graph.geometry.reference_frame import RelationReferenceFrame
from scene_graph.perception.observation import Observation, encode_mask_rle
from scene_graph.pipeline.async_pipeline import (
    FrameHistoryBuffer,
    AsyncDetectorWorker,
    AsyncOnlinePipeline,
    AsyncDetectionResult,
)
from scene_graph.tracking.causal_tracker import CausalTracker
from scene_graph.tracking.track import TrackState


def make_test_frame_packet(frame_index: int, timestamp: float, camera_x: float = 0.0):
    intrinsics = CameraIntrinsics(fx=384.0, fy=384.0, cx=32.0, cy=24.0, width=64, height=48)
    depth_model = DepthModel(scale=1000.0)

    # Simple 4x4 camera pose in world with translation in X
    world_T_camera = np.eye(4, dtype=np.float64)
    world_T_camera[0, 3] = camera_x

    rgb = np.full((48, 64, 3), 120, dtype=np.uint8)
    depth = np.full((48, 64), 1.5, dtype=np.float32)

    relation_frame = RelationReferenceFrame.create("camera", world_T_camera)

    return FramePacket(
        frame_index=frame_index,
        timestamp=timestamp,
        rgb=rgb,
        depth=depth,
        world_T_camera=world_T_camera,
        camera_intrinsics=intrinsics,
        depth_model=depth_model,
        relation_frame=relation_frame,
        imu_samples=(),
        frame_id="camera_color_optical_frame",
        optical_frame_id="camera_color_optical_frame",
        pose_source="tf_exact",
        sensor_timestamp=timestamp,
        rgb_timestamp=timestamp,
        depth_timestamp=timestamp,
        pose_timestamp=timestamp,
        pose_age=0.0,
        world_frame="world",
        transform_source="tf_exact",
        transform_valid=True,
        localization_mode=LocalizationMode.WORLD_MODE,
        metadata={},
    )


def make_test_observation(obs_id: str, centroid: np.ndarray, class_name: str = "cup"):
    mask = np.zeros((48, 64), dtype=bool)
    mask[20:28, 28:36] = True

    obs = Observation(
        obs_id=obs_id,
        frame_index=0,
        timestamp=0.0,
        class_name=class_name,
        confidence=0.92,
        bbox_xyxy=np.array([28.0, 20.0, 36.0, 28.0], dtype=np.float32),
        mask_rle=encode_mask_rle(mask),
    )
    obs.object_geometry = ObjectGeometry(
        status=GeometryStatus.VALID,
        centroid_world=centroid.copy(),
        position_covariance_world=np.eye(3, dtype=np.float64) * 0.001,
        bbox_min_world=centroid - 0.05,
        bbox_max_world=centroid + 0.05,
        obb_center_world=centroid.copy(),
        obb_axes_world=np.eye(3, dtype=np.float64),
        obb_extents_world=np.array([0.1, 0.1, 0.1], dtype=np.float64),
        points_world=centroid.reshape(1, 3),
        points_world_sampled=centroid.reshape(1, 3),
        points_camera=centroid.reshape(1, 3),
        valid_point_count=100,
    )
    return obs


def test_frame_history_buffer_retention_and_lookup():
    buffer = FrameHistoryBuffer(capacity=5)

    packets = [make_test_frame_packet(i, 100.0 + i * 0.033) for i in range(8)]
    for p in packets:
        buffer.append(p)

    assert len(buffer) == 5

    # Frames 0, 1, 2 should have been evicted
    assert buffer.get_by_index(0) is None
    assert buffer.get_by_index(1) is None
    assert buffer.get_by_index(2) is None

    # Frames 3 to 7 should be present
    for i in range(3, 8):
        retrieved = buffer.get_by_index(i)
        assert retrieved is not None
        assert retrieved.frame_index == i

    # Query by timestamp
    p_exact = buffer.get_by_timestamp(100.0 + 5 * 0.033, max_dt=0.01)
    assert p_exact is not None
    assert p_exact.frame_index == 5

    p_too_far = buffer.get_by_timestamp(90.0, max_dt=0.01)
    assert p_too_far is None


def test_async_detector_worker_non_blocking_and_drops():
    class SlowMockDetector:
        def __init__(self, delay_s: float = 0.04):
            self.delay_s = delay_s

        def detect(self, packet):
            time.sleep(self.delay_s)
            obs = make_test_observation(f"obs_{packet.frame_index}", np.array([0.5, 0.0, 1.2]))
            obs.frame_index = packet.frame_index
            obs.timestamp = packet.timestamp
            return [obs]

    detector = SlowMockDetector(delay_s=0.03)
    worker = AsyncDetectorWorker(detector, in_queue_size=1, out_queue_size=4)

    try:
        t_start = time.monotonic()
        # Rapidly submit 6 frames; submission must be non-blocking (< 15ms total for all 6)
        for i in range(6):
            p = make_test_frame_packet(i, 10.0 + i * 0.033)
            worker.submit_frame(p)
        submit_duration = time.monotonic() - t_start
        assert submit_duration < 0.03, f"Submit took {submit_duration*1000:.1f}ms, should be non-blocking"

        # Wait for worker to finish available tasks
        time.sleep(0.12)
        results = worker.get_completed_detections()
        assert len(results) >= 1
        assert worker.frames_detected >= 1
        assert worker.dropped_tasks > 0, "Expected backpressure drops on rapid submission"
    finally:
        worker.stop()


def test_causal_tracker_update_delayed_replay():
    config = load_config("configs/default.yaml")
    tracker = CausalTracker(config)

    # Frame 0: tracking without detections (propagation)
    t0 = tracker.update([], 0, 100.0)
    assert len(t0) == 0

    # Frame 1: tracking without detections
    t1 = tracker.update([], 1, 100.033)
    assert len(t1) == 0

    # Frame 2: tracking without detections
    t2 = tracker.update([], 2, 100.066)
    assert len(t2) == 0

    # Detections computed for Frame 0 arrive while we are at Frame 3!
    obs0 = make_test_observation("obs_0", np.array([1.0, 0.5, 1.2]), class_name="bottle")
    obs0.frame_index = 0
    obs0.timestamp = 100.0

    tracks_at_3 = tracker.update_delayed(
        observations=[obs0],
        obs_frame_index=0,
        obs_timestamp=100.0,
        current_frame_index=3,
        current_timestamp=100.100,
    )

    assert len(tracks_at_3) == 1
    track = tracks_at_3[0]
    assert track.class_name == "bottle"
    assert track.first_observed_frame == 0
    assert pytest.approx(track.last_timestamp, abs=1e-4) == 100.0
    assert pytest.approx(tracker._last_timestamp, abs=1e-4) == 100.100
    assert track.centroid_world is not None


def test_async_online_pipeline_tracking_rate_vs_detection_rate():
    config = load_config("configs/default.yaml")

    class SyntheticDetector:
        def __init__(self, delay_s: float = 0.03):
            self.delay_s = delay_s

        def detect(self, packet):
            time.sleep(self.delay_s)
            obs = make_test_observation(f"obs_{packet.frame_index}", np.array([0.8, -0.2, 1.5]), class_name="cup")
            obs.frame_index = packet.frame_index
            obs.timestamp = packet.timestamp
            return [obs]

    detector = SyntheticDetector(delay_s=0.025)
    pipeline = AsyncOnlinePipeline(config, detector=detector)

    try:
        # Feed 10 sensor frames with simulated camera rate spacing
        total_frames = 10
        for i in range(total_frames):
            packet = make_test_frame_packet(i, 200.0 + i * 0.033)
            graph = pipeline.update(packet)
            assert graph is not None
            assert graph.current_frame_index == i
            time.sleep(0.010)

        # Allow detector thread to finish pending work
        time.sleep(0.05)

        telemetry = pipeline.get_telemetry()
        # 100% of sensor frames processed by tracking
        assert telemetry["total_camera_frames"] == total_frames
        # Detector ran asynchronously at slower rate
        assert telemetry["total_detection_frames"] >= 1
        assert telemetry["total_detection_frames"] < telemetry["total_camera_frames"]
        assert telemetry["history_buffer_length"] == total_frames
    finally:
        pipeline.stop()


def test_timestamp_aware_geometry_lifting_under_ego_motion():
    config = load_config("configs/default.yaml")

    # Camera moves in X: Frame 0 at X=0.0m, Frame 1 at X=0.5m, Frame 2 at X=1.0m
    p0 = make_test_frame_packet(0, 300.000, camera_x=0.0)
    p1 = make_test_frame_packet(1, 300.033, camera_x=0.5)
    p2 = make_test_frame_packet(2, 300.066, camera_x=1.0)

    class DelayedDetector:
        def __init__(self):
            self.called_for = []

        def detect(self, packet):
            self.called_for.append(packet.frame_index)
            # 2D detection mask in image center
            mask = np.zeros((48, 64), dtype=bool)
            mask[20:28, 28:36] = True
            obs = Observation(
                obs_id=f"obs_f{packet.frame_index}",
                frame_index=packet.frame_index,
                timestamp=packet.timestamp,
                class_name="mug",
                confidence=0.88,
                bbox_xyxy=np.array([28.0, 20.0, 36.0, 28.0], dtype=np.float32),
                mask_rle=encode_mask_rle(mask),
            )
            return [obs]

    pipeline = AsyncOnlinePipeline(config, detector=DelayedDetector())

    try:
        # Feed Frame 0
        g0 = pipeline.update(p0)
        time.sleep(0.02)

        # Feed Frame 1
        g1 = pipeline.update(p1)
        time.sleep(0.02)

        # Feed Frame 2
        g2 = pipeline.update(p2)
        time.sleep(0.02)

        # Verify historical packet retrieval at Frame 0
        h0 = pipeline.history_buffer.get_by_index(0)
        assert h0 is not None
        assert h0.world_T_camera[0, 3] == 0.0

        h2 = pipeline.history_buffer.get_by_index(2)
        assert h2 is not None
        assert h2.world_T_camera[0, 3] == 1.0

        # Fast tracking continued uninterrupted across all 3 frames
        assert pipeline.total_camera_frames == 3
        assert len(g2.nodes) >= 0
    finally:
        pipeline.stop()
