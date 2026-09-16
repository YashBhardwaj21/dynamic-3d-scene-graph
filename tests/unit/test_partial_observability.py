import numpy as np
import pytest

from scene_graph.config import RelationTemporalConfig, SceneGraphConfig
from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.point_cloud import ObjectGeometry, GeometryStatus
from scene_graph.geometry.provenance import GeometrySource
from scene_graph.geometry.reference_frame import CameraFrame, RelationReferenceFrame
from scene_graph.perception.observation import Observation, encode_mask_rle
from scene_graph.pipeline.pipeline_core import SceneGraphPipeline
from scene_graph.relations.context import FrameContext, ObservationGeometry
from scene_graph.relations.evidence import EvidenceResult, ReferenceFrameType, RelationEvidence
from scene_graph.relations.occlusion import OcclusionRelationModule
from scene_graph.temporal.relation_state import RelationState, RelationStateMachine
from scene_graph.tracking.track import Track, TrackState


def test_unobserved_track_predicted_geometry():
    """Asserts active track without detection in current frame receives GeometrySource.PREDICTED."""
    config = SceneGraphConfig.from_files("configs/default.yaml")
    pipeline = SceneGraphPipeline(config)

    # Create dummy observation at frame 0
    obs_geom = ObjectGeometry(
        status=GeometryStatus.VALID,
        centroid_world=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        position_covariance_world=np.eye(3, dtype=np.float64) * 0.01,
        bbox_min_world=np.array([0.8, 1.8, 2.8], dtype=np.float64),
        bbox_max_world=np.array([1.2, 2.2, 3.2], dtype=np.float64),
        obb_center_world=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        obb_axes_world=np.eye(3, dtype=np.float64),
        obb_extents_world=np.array([0.4, 0.4, 0.4], dtype=np.float64),
        depth_stats={"mean": 3.0},
        points_world_sampled=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        points_world=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        points_camera=np.array([[0.0, 0.0, 3.0]], dtype=np.float64),
        valid_point_count=100,
    )

    mask = np.ones((100, 100), dtype=bool)
    obs = Observation(
        obs_id="obs_0001",
        frame_index=0,
        timestamp=0.0,
        class_name="cup",
        confidence=0.9,
        bbox_xyxy=np.array([10.0, 10.0, 50.0, 50.0]),
        mask_rle=encode_mask_rle(mask),
        object_geometry=obs_geom,
    )

    from collections import deque
    track = Track(
        object_id="track_0001",
        class_name="cup",
        state=TrackState.TEMPORARILY_UNOBSERVED,
        _initial_centroid=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        last_observed_frame=0,
        first_observed_frame=0,
        observation_count=1,
        missing_count=1,
        detection_confidence=0.9,
        track_observation_ratio=0.5,
        last_timestamp=0.0,
        recent_observations=deque([obs]),
    )

    # Frame 1: track is unobserved (observation is from frame 0)
    geometries = pipeline._build_observation_geometry([track], frame_index=1)

    assert "track_0001" in geometries
    predicted_geom = geometries["track_0001"]
    assert predicted_geom.geometry_source == GeometrySource.PREDICTED
    assert np.allclose(predicted_geom.centroid_world, [1.0, 2.0, 3.0])
    assert predicted_geom.mask is None
    assert predicted_geom.points_camera is None


def test_relation_decay_over_occlusion_window():
    """Asserts relation decays gracefully over occlusion window rather than deleting on frame 1."""
    config = RelationTemporalConfig(
        confirmation_threshold=0.6,
        contradiction_threshold=-0.5,
        decay_per_second=0.2,
        unknown_after_seconds=1.0,
        lost_after_seconds=2.5,
    )
    machine = RelationStateMachine(config)

    # Frame 0: Provide positive evidence
    ev = RelationEvidence(
        predicate="ON",
        subject_id="obj_A",
        object_id="obj_B",
        frame_index=0,
        timestamp=0.0,
        result=EvidenceResult.SUPPORTED,
        value=0.9,
        threshold=0.5,
        confidence=0.9,
        reference_frame=ReferenceFrameType.WORLD,
        evidence_type="support",
        details={},
    )
    states = machine.update([ev], frame_index=0, timestamp=0.0)
    key = ("obj_A", "obj_B", "ON")
    assert states[key] == RelationState.SUPPORTED

    # Frame 1: 0.2s later, object occluded, no evidence provided
    # Crucially, relation must NOT be deleted!
    states = machine.update([], frame_index=1, timestamp=0.2, active_object_ids={"obj_A", "obj_B"})
    assert key in states
    assert states[key] == RelationState.SUPPORTED

    # Frame 2: 1.2s later (missing_time = 1.2s >= unknown_after_seconds = 1.0s)
    # Relation should transition to UNKNOWN, not yet deleted
    states = machine.update([], frame_index=2, timestamp=1.2, active_object_ids={"obj_A", "obj_B"})
    assert key in states
    assert states[key] == RelationState.UNKNOWN

    # Frame 3: 2.6s later (missing_time = 2.6s >= lost_after_seconds = 2.5s)
    # Relation is purged
    states = machine.update([], frame_index=3, timestamp=2.6, active_object_ids={"obj_A", "obj_B"})
    assert key not in states


def test_occlusion_module_rejects_predicted_geometry():
    """Asserts OcclusionRelationModule refuses to compute on predicted geometry."""
    cfg = SceneGraphConfig.from_files("configs/default.yaml")
    module = OcclusionRelationModule(cfg)

    obs_geom = ObservationGeometry(
        obs_id="obs_0001",
        track_id="track_0001",
        centroid_world=np.array([0.0, 0.0, 1.0]),
        bbox_min_world=np.array([-0.1, -0.1, 0.9]),
        bbox_max_world=np.array([0.1, 0.1, 1.1]),
        depth_stats={"mean": 1.0},
        points_world_sampled=None,
        points_world=None,
        points_camera=np.array([[0.0, 0.0, 1.0]]),
        mask=np.ones((50, 50), dtype=bool),
        valid_point_count=50,
        geometry_source=GeometrySource.PREDICTED,
    )

    track_a = Track(
        object_id="track_0001",
        class_name="cup",
        state=TrackState.ACTIVE,
        _initial_centroid=np.zeros(3),
        last_observed_frame=0,
        first_observed_frame=0,
        observation_count=1,
        missing_count=0,
        detection_confidence=0.9,
        track_observation_ratio=1.0,
        last_timestamp=0.0,
        recent_observations=[],
    )
    track_b = Track(
        object_id="track_0002",
        class_name="bottle",
        state=TrackState.ACTIVE,
        _initial_centroid=np.zeros(3),
        last_observed_frame=0,
        first_observed_frame=0,
        observation_count=1,
        missing_count=0,
        detection_confidence=0.9,
        track_observation_ratio=1.0,
        last_timestamp=0.0,
        recent_observations=[],
    )

    ref_frame = RelationReferenceFrame(
        origin_world=np.zeros(3),
        up_axis_world=np.array([0.0, 1.0, 0.0]),
        horizontal_axis_world=np.array([1.0, 0.0, 0.0]),
        depth_axis_world=np.array([0.0, 0.0, 1.0]),
    )

    ctx = FrameContext(
        frame_index=0,
        timestamp=0.0,
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=25.0, cy=25.0, width=50, height=50),
        world_T_camera=np.eye(4),
        reference_frame=ref_frame,
        camera_frame=CameraFrame.from_camera_pose(np.eye(4)),
        depth_image=np.ones((50, 50), dtype=np.float32),
        observation_geometry={"track_0001": obs_geom, "track_0002": obs_geom},
    )

    results = module.compute(track_a, track_b, ctx)
    assert len(results) == 0


def test_active_object_ids_filters_output_without_destroying_history():
    """Verify active_object_ids excludes lost tracks from current output while preserving internal belief decay."""
    config = RelationTemporalConfig(
        confirmation_threshold=0.6,
        contradiction_threshold=-0.5,
        decay_per_second=0.1,
        unknown_after_seconds=2.0,
        lost_after_seconds=5.0,
    )
    machine = RelationStateMachine(config)

    ev = RelationEvidence(
        predicate="ON",
        subject_id="obj_A",
        object_id="obj_B",
        frame_index=0,
        timestamp=0.0,
        result=EvidenceResult.SUPPORTED,
        value=0.9,
        threshold=0.5,
        confidence=0.9,
        reference_frame=ReferenceFrameType.WORLD,
        evidence_type="support",
        details={},
    )
    key = ("obj_A", "obj_B", "ON")

    # Frame 0: Both active
    states = machine.update([ev], frame_index=0, timestamp=0.0, active_object_ids={"obj_A", "obj_B"})
    assert key in states
    assert states[key] == RelationState.SUPPORTED

    # Frame 1: obj_B is lost (only obj_A is active)
    # Output must NOT contain the relation
    states = machine.update([], frame_index=1, timestamp=0.5, active_object_ids={"obj_A"})
    assert key not in states
    # But internal belief and state must still exist and decay
    assert key in machine.states
    assert key in machine.beliefs
    assert machine.beliefs[key] > 0.0

    # Frame 2: obj_B reappears before timeout
    states = machine.update([], frame_index=2, timestamp=1.0, active_object_ids={"obj_A", "obj_B"})
    assert key in states
    assert states[key] == RelationState.SUPPORTED

