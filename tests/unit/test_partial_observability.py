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


def test_frustum_math():
    """Verify camera frustum boundary math, clipping, and box containment."""
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    world_T_cam = np.eye(4, dtype=np.float64)

    # 1. Point directly in front of camera inside FOV
    pt_in = np.array([0.0, 0.0, 2.0])
    assert intrinsics.is_in_frustum(pt_in) is True
    assert intrinsics.is_world_point_in_frustum(pt_in, world_T_cam) is True

    # 2. Point behind camera (Z <= 0)
    pt_behind = np.array([0.0, 0.0, -1.0])
    assert intrinsics.is_in_frustum(pt_behind) is False
    assert intrinsics.is_world_point_in_frustum(pt_behind, world_T_cam) is False

    # 3. Point too close (Z < min_depth_m)
    pt_too_close = np.array([0.0, 0.0, 0.05])
    assert intrinsics.is_in_frustum(pt_too_close, min_depth_m=0.10) is False

    # 4. Point too far (Z > max_depth_m)
    pt_too_far = np.array([0.0, 0.0, 15.0])
    assert intrinsics.is_in_frustum(pt_too_far, max_depth_m=10.0) is False

    # 5. Point laterally outside horizontal FOV
    # u = (x * 500 / 2.0) + 320 = 250*x + 320. If x = 2.0, u = 820 > 640.
    pt_wide = np.array([2.0, 0.0, 2.0])
    assert intrinsics.is_in_frustum(pt_wide) is False

    # 6. Vectorized points check
    pts = np.array([
        [0.0, 0.0, 2.0],
        [0.0, 0.0, -1.0],
        [2.0, 0.0, 2.0],
        [0.0, 0.0, 1.0],
    ])
    mask = intrinsics.points_in_frustum(pts)
    assert np.array_equal(mask, [True, False, False, True])

    # 7. Box in frustum
    bbox_min = np.array([-0.1, -0.1, 1.9])
    bbox_max = np.array([0.1, 0.1, 2.1])
    assert intrinsics.is_box_in_frustum(bbox_min, bbox_max, world_T_cam) is True

    bbox_behind_min = np.array([-0.1, -0.1, -3.0])
    bbox_behind_max = np.array([0.1, 0.1, -2.0])
    assert intrinsics.is_box_in_frustum(bbox_behind_min, bbox_behind_max, world_T_cam) is False


def test_out_of_view_vs_occluded_lifetime():
    """Verify that out-of-view tracks persist longer than occluded tracks inside frustum."""
    from scene_graph.tracking.causal_tracker import CausalTracker
    from scene_graph.ontology.entity import VisibilityState

    config = SceneGraphConfig.from_files("configs/default.yaml")
    tracker = CausalTracker(config)
    tracker.max_missing_seconds = 1.0
    tracker.max_missing_seconds_out_of_view = 5.0
    assert tracker.max_missing_seconds == 1.0
    assert tracker.max_missing_seconds_out_of_view >= 5.0

    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    world_T_cam = np.eye(4, dtype=np.float64)

    def make_obs(centroid, class_name, obs_id):
        geom = ObjectGeometry(
            status=GeometryStatus.VALID,
            centroid_world=np.asarray(centroid, dtype=np.float64),
            position_covariance_world=np.eye(3) * 0.01,
            bbox_min_world=np.asarray(centroid) - 0.1,
            bbox_max_world=np.asarray(centroid) + 0.1,
            obb_center_world=np.asarray(centroid),
            obb_axes_world=np.eye(3),
            obb_extents_world=np.array([0.2, 0.2, 0.2]),
            depth_stats={"mean": float(centroid[2])},
            points_world_sampled=np.array([centroid]),
            points_world=np.array([centroid]),
            points_camera=np.array([centroid]),
            valid_point_count=50,
        )
        return Observation(
            obs_id=obs_id,
            frame_index=0,
            timestamp=0.0,
            class_name=class_name,
            confidence=0.9,
            bbox_xyxy=np.array([10.0, 10.0, 50.0, 50.0]),
            mask_rle=encode_mask_rle(np.ones((10, 10), dtype=bool)),
            object_geometry=geom,
        )

    # Initialize two tracks over 3 frames to CONFIRM them
    # Obj A at [0, 0, 2.0] (in front of camera)
    # Obj B at [0, 0, -2.0] (behind camera in world frame)
    for fi in range(3):
        t = fi * 0.033
        obs_a = make_obs([0.0, 0.0, 2.0], "cup", f"obs_a_{fi}")
        obs_b = make_obs([0.0, 0.0, -2.0], "bottle", f"obs_b_{fi}")
        tracker.update([obs_a, obs_b], fi, t, camera_intrinsics=intrinsics, world_T_camera=world_T_cam)

    assert tracker.tracks["track_0001"].state == TrackState.ACTIVE
    assert tracker.tracks["track_0002"].state == TrackState.ACTIVE

    # Frame 3 (t = 0.5s): Neither is observed
    tracks = tracker.update([], 3, 0.5, camera_intrinsics=intrinsics, world_T_camera=world_T_cam)
    track_a = tracker.tracks["track_0001"]
    track_b = tracker.tracks["track_0002"]

    assert track_a.state == TrackState.TEMPORARILY_UNOBSERVED
    assert track_a.is_in_frustum is True
    assert track_a.visibility_state == VisibilityState.OCCLUDED

    assert track_b.state == TrackState.TEMPORARILY_UNOBSERVED
    assert track_b.is_in_frustum is False
    assert track_b.visibility_state == VisibilityState.OUT_OF_VIEW

    # Frame 4 (t = 1.5s): 1.5s since last observation at t=0.066
    # Track A is in frustum -> missing_seconds >= 1.0s -> LOST
    # Track B is out of view -> missing_seconds < 5.0s -> STILL TEMPORARILY_UNOBSERVED!
    tracks = tracker.update([], 4, 1.5, camera_intrinsics=intrinsics, world_T_camera=world_T_cam)
    assert "track_0001" not in tracker.tracks or tracker.tracks["track_0001"].state == TrackState.LOST
    assert tracker.tracks["track_0002"].state == TrackState.TEMPORARILY_UNOBSERVED
    assert tracker.tracks["track_0002"].visibility_state == VisibilityState.OUT_OF_VIEW

    # Frame 5 (t = 5.5s): missing_seconds >= 5.0s -> Track B now LOST
    tracks = tracker.update([], 5, 5.5, camera_intrinsics=intrinsics, world_T_camera=world_T_cam)
    assert "track_0002" not in tracker.tracks or tracker.tracks["track_0002"].state == TrackState.LOST


def test_out_of_view_reassociation():
    """Verify that an out-of-view object re-associates to its original track when camera turns back."""
    from scene_graph.tracking.causal_tracker import CausalTracker

    config = SceneGraphConfig.from_files("configs/default.yaml")
    tracker = CausalTracker(config)

    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    world_T_cam_forward = np.eye(4, dtype=np.float64)

    # Camera looking away: 180 deg yaw rotation
    world_T_cam_away = np.array([
        [-1.0,  0.0,  0.0, 0.0],
        [ 0.0,  1.0,  0.0, 0.0],
        [ 0.0,  0.0, -1.0, 0.0],
        [ 0.0,  0.0,  0.0, 1.0],
    ], dtype=np.float64)

    def make_obs(centroid, obs_id):
        geom = ObjectGeometry(
            status=GeometryStatus.VALID,
            centroid_world=np.asarray(centroid, dtype=np.float64),
            position_covariance_world=np.eye(3) * 0.01,
            bbox_min_world=np.asarray(centroid) - 0.1,
            bbox_max_world=np.asarray(centroid) + 0.1,
            obb_center_world=np.asarray(centroid),
            obb_axes_world=np.eye(3),
            obb_extents_world=np.array([0.2, 0.2, 0.2]),
            depth_stats={"mean": float(centroid[2])},
            points_world_sampled=np.array([centroid]),
            points_world=np.array([centroid]),
            points_camera=np.array([centroid]),
            valid_point_count=50,
        )
        return Observation(
            obs_id=obs_id,
            frame_index=0,
            timestamp=0.0,
            class_name="cup",
            confidence=0.9,
            bbox_xyxy=np.array([10.0, 10.0, 50.0, 50.0]),
            mask_rle=encode_mask_rle(np.ones((10, 10), dtype=bool)),
            object_geometry=geom,
        )

    # 1. Observe object at [0, 0, 2.0] for 3 frames to confirm
    for fi in range(3):
        t = fi * 0.033
        obs = make_obs([0.0, 0.0, 2.0], f"obs_{fi}")
        tracker.update([obs], fi, t, camera_intrinsics=intrinsics, world_T_camera=world_T_cam_forward)

    orig_track_id = "track_0001"
    assert tracker.tracks[orig_track_id].state == TrackState.ACTIVE

    # 2. Camera turns away at t = 0.5s and t = 2.0s (object is out of view for 2.0s > 1.0s)
    tracker.update([], 3, 0.5, camera_intrinsics=intrinsics, world_T_camera=world_T_cam_away)
    tracker.update([], 4, 2.0, camera_intrinsics=intrinsics, world_T_camera=world_T_cam_away)

    assert orig_track_id in tracker.tracks
    assert tracker.tracks[orig_track_id].state == TrackState.TEMPORARILY_UNOBSERVED
    assert tracker.tracks[orig_track_id].is_in_frustum is False

    # 3. Camera turns back at t = 2.1s and observes the object again
    obs_reappear = make_obs([0.0, 0.0, 2.0], "obs_reappear")
    tracks = tracker.update([obs_reappear], 5, 2.1, camera_intrinsics=intrinsics, world_T_camera=world_T_cam_forward)

    # Must re-associate to orig_track_id, NOT create track_0002!
    assert len(tracker.tracks) == 1
    assert orig_track_id in tracker.tracks
    assert tracker.tracks[orig_track_id].state == TrackState.ACTIVE
    assert tracker.tracks[orig_track_id].is_in_frustum is True


def test_pipeline_persistent_entity_out_of_view_visibility():
    """Verify SceneGraphPipeline populates VisibilityState.OUT_OF_VIEW on nodes and entities."""
    from scene_graph.data.frame_packet import FramePacket
    from scene_graph.ontology.entity import VisibilityState

    config = SceneGraphConfig.from_files("configs/default.yaml")
    pipeline = SceneGraphPipeline(config)

    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    world_T_cam_forward = np.eye(4, dtype=np.float64)
    world_T_cam_away = np.array([
        [-1.0,  0.0,  0.0, 0.0],
        [ 0.0,  1.0,  0.0, 0.0],
        [ 0.0,  0.0, -1.0, 0.0],
        [ 0.0,  0.0,  0.0, 1.0],
    ], dtype=np.float64)

    def make_packet_and_obs(frame_idx, timestamp, world_T_cam, with_obs=True):
        pkt = FramePacket(
            frame_index=frame_idx,
            timestamp=timestamp,
            rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            depth=np.ones((480, 640), dtype=np.float32) * 2.0,
            camera_intrinsics=intrinsics,
            world_T_camera=world_T_cam,
            relation_frame=RelationReferenceFrame.create("map", np.eye(4)),
        )
        if not with_obs:
            return pkt, []

        geom = ObjectGeometry(
            status=GeometryStatus.VALID,
            centroid_world=np.array([0.0, 0.0, 2.0]),
            position_covariance_world=np.eye(3) * 0.01,
            bbox_min_world=np.array([-0.1, -0.1, 1.9]),
            bbox_max_world=np.array([0.1, 0.1, 2.1]),
            obb_center_world=np.array([0.0, 0.0, 2.0]),
            obb_axes_world=np.eye(3),
            obb_extents_world=np.array([0.2, 0.2, 0.2]),
            depth_stats={"mean": 2.0},
            points_world_sampled=np.array([[0.0, 0.0, 2.0]]),
            points_world=np.array([[0.0, 0.0, 2.0]]),
            points_camera=np.array([[0.0, 0.0, 2.0]]),
            valid_point_count=50,
        )
        obs = Observation(
            obs_id=f"obs_{frame_idx}",
            frame_index=frame_idx,
            timestamp=timestamp,
            class_name="cup",
            confidence=0.9,
            bbox_xyxy=np.array([10.0, 10.0, 50.0, 50.0]),
            mask_rle=encode_mask_rle(np.ones((10, 10), dtype=bool)),
            object_geometry=geom,
        )
        return pkt, [obs]

    # Confirm track over 3 frames
    for fi in range(3):
        pkt, obs_list = make_packet_and_obs(fi, fi * 0.033, world_T_cam_forward, with_obs=True)
        graph = pipeline.update(pkt, obs_list)

    assert "track_0001" in graph.nodes
    node = graph.nodes["track_0001"]
    assert node.attributes["visibility"] == "observed"
    assert node.attributes["in_frustum"] is True
    entity = node.to_persistent_entity()
    assert entity.visibility_state == VisibilityState.VISIBLE

    # Frame 3: Camera turns away, object is unobserved and out of view
    pkt_away, obs_away = make_packet_and_obs(3, 0.20, world_T_cam_away, with_obs=False)
    graph = pipeline.update(pkt_away, obs_away)

    assert "track_0001" in graph.nodes
    node = graph.nodes["track_0001"]
    assert node.attributes["visibility"] == "out_of_view"
    assert node.attributes["in_frustum"] is False
    entity = node.to_persistent_entity()
    assert entity.visibility_state == VisibilityState.OUT_OF_VIEW

    # Frame 4: Camera turns back, but object is occluded (inside frustum, no detection)
    pkt_back, obs_back = make_packet_and_obs(4, 0.40, world_T_cam_forward, with_obs=False)
    graph = pipeline.update(pkt_back, obs_back)

    assert "track_0001" in graph.nodes
    node = graph.nodes["track_0001"]
    assert node.attributes["visibility"] == "predicted"
    assert node.attributes["visibility_state"] == "occluded"
    assert node.attributes["in_frustum"] is True
    entity = node.to_persistent_entity()
    assert entity.visibility_state == VisibilityState.OCCLUDED

