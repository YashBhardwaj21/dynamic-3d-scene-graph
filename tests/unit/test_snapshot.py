"""Tests for SceneGraphSnapshot creation and immutability."""

import numpy as np
import pytest
from collections import deque

from scene_graph.config import SceneGraphConfig
from scene_graph.data.frame_packet import FramePacket
from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.geometry.reference_frame import RelationReferenceFrame
from scene_graph.geometry.point_cloud import ObjectGeometry, GeometryStatus
from scene_graph.graph.temporal_graph import TemporalSceneGraph
from scene_graph.graph.node import GraphNode
from scene_graph.graph.edge import GraphEdge
from scene_graph.graph.participation_state import GraphParticipationState
from scene_graph.graph.snapshot import (
    SceneGraphSnapshot,
    ObjectSnapshot,
    RelationSnapshot,
    TelemetrySnapshot,
    create_snapshot,
)
from scene_graph.tracking.track import Track, TrackState
from scene_graph.tracking.state import KalmanState
from scene_graph.temporal.object_state import ObjectState
from scene_graph.temporal.relation_state import RelationState
from scene_graph.relations.evidence import RelationEvidence, EvidenceResult, ReferenceFrameType
from scene_graph.perception.observation import Observation


def _make_track(track_id, class_name, centroid, state=TrackState.ACTIVE):
    """Create a minimal Track with a KalmanState."""
    centroid = np.asarray(centroid, dtype=np.float64)
    obs = Observation(
        obs_id=f"obs_{track_id}",
        frame_index=10,
        timestamp=1.0,
        class_name=class_name,
        confidence=0.85,
        bbox_xyxy=np.zeros(4),
        mask_rle=None,
    )
    obs.object_geometry = ObjectGeometry(
        centroid_world=centroid.copy(),
        centroid_camera=centroid.copy(),
        bbox_min_world=centroid - 0.1,
        bbox_max_world=centroid + 0.1,
        obb_center_world=centroid.copy(),
        obb_axes_world=np.eye(3),
        obb_extents_world=np.array([0.2, 0.2, 0.2]),
        valid_point_count=100,
        status=GeometryStatus.VALID,
    )
    return Track(
        object_id=track_id,
        class_name=class_name,
        state=state,
        _initial_centroid=centroid.copy(),
        last_observed_frame=10,
        first_observed_frame=1,
        observation_count=5,
        missing_count=0,
        detection_confidence=0.85,
        track_observation_ratio=1.0,
        last_timestamp=1.0,
        recent_observations=deque([obs], maxlen=10),
        kalman_state=KalmanState(centroid.copy(), initial_cov_pos=1.0, initial_cov_vel=1.0),
    )


def _make_packet(frame_index=10, timestamp=1.0):
    """Create a minimal FramePacket."""
    return FramePacket(
        frame_index=frame_index,
        timestamp=timestamp,
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        depth=np.zeros((480, 640), dtype=np.float32),
        world_T_camera=np.eye(4),
        camera_intrinsics=CameraIntrinsics(fx=525, fy=525, cx=319.5, cy=239.5, width=640, height=480),
        depth_model=DepthModel(scale=5000.0),
        relation_frame=RelationReferenceFrame.from_gravity_and_heading(
            np.zeros(3), np.array([0, 0, 1.0]), np.array([1, 0, 0.0]),
        ),
    )


def _build_test_graph():
    """Build a TemporalSceneGraph with 2 objects and 1 relation for testing."""
    graph = TemporalSceneGraph()
    graph.current_frame_index = 10
    graph.current_timestamp = 1.0

    track_a = _make_track("track_0001", "monitor", [1.0, 0.0, 0.5])
    track_b = _make_track("track_0002", "keyboard", [1.0, 0.0, 0.0])

    graph.nodes["track_0001"] = GraphNode(
        object_id="track_0001",
        class_name="monitor",
        track=track_a,
        state=ObjectState.STABLE,
    )
    graph.nodes["track_0002"] = GraphNode(
        object_id="track_0002",
        class_name="keyboard",
        track=track_b,
        state=ObjectState.STABLE,
    )

    evidence = RelationEvidence(
        predicate="ON",
        subject_id="track_0002",
        object_id="track_0001",
        frame_index=10,
        timestamp=1.0,
        result=EvidenceResult.SUPPORTED,
        value=0.05,
        threshold=0.1,
        confidence=0.9,
        reference_frame=ReferenceFrameType.WORLD,
        evidence_type="support",
        details={},
    )
    graph.edges[("track_0002", "track_0001", "ON")] = GraphEdge(
        predicate="ON",
        subject_id="track_0002",
        object_id="track_0001",
        state=RelationState.SUPPORTED,
        latest_evidence=evidence,
        participation=GraphParticipationState.ACTIVE,
    )

    return graph


class TestSceneGraphSnapshot:

    def test_snapshot_creation(self):
        """create_snapshot produces a valid SceneGraphSnapshot."""
        graph = _build_test_graph()
        packet = _make_packet()

        snap = create_snapshot(graph, packet, processing_fps=0.6, queue_size=5)

        assert isinstance(snap, SceneGraphSnapshot)
        assert snap.frame_index == 10
        assert snap.timestamp == 1.0
        assert len(snap.objects) == 2
        assert len(snap.relations) == 1

    def test_snapshot_is_frozen(self):
        """Snapshot dataclasses are immutable."""
        graph = _build_test_graph()
        packet = _make_packet()
        snap = create_snapshot(graph, packet)

        with pytest.raises(AttributeError):
            snap.frame_index = 999

        with pytest.raises(AttributeError):
            snap.objects[0].track_id = "hacked"

    def test_object_snapshot_fields(self):
        """ObjectSnapshot contains correct field values from the track."""
        graph = _build_test_graph()
        packet = _make_packet()
        snap = create_snapshot(graph, packet)

        monitor = [o for o in snap.objects if o.class_name == "monitor"][0]
        assert monitor.track_id == "track_0001"
        assert monitor.state == "active"
        assert monitor.object_state == "stable"
        assert monitor.confidence == pytest.approx(0.85)
        assert monitor.observation_count == 5
        assert monitor.centroid_world is not None
        assert len(monitor.centroid_world) == 3
        assert monitor.obb_center_world is not None
        assert monitor.obb_axes_world is not None
        assert monitor.obb_extents_world is not None

    def test_relation_snapshot_fields(self):
        """RelationSnapshot contains correct field values from the edge."""
        graph = _build_test_graph()
        packet = _make_packet()
        snap = create_snapshot(graph, packet)

        assert len(snap.relations) == 1
        rel = snap.relations[0]
        assert rel.subject_id == "track_0002"
        assert rel.subject_class == "keyboard"
        assert rel.predicate == "ON"
        assert rel.object_id == "track_0001"
        assert rel.object_class == "monitor"
        assert rel.state == "supported"
        assert rel.confidence == pytest.approx(0.9)

    def test_telemetry_fields(self):
        """TelemetrySnapshot captures provided metrics."""
        graph = _build_test_graph()
        packet = _make_packet()
        snap = create_snapshot(
            graph, packet,
            input_fps=30.0,
            processing_fps=0.6,
            total_latency_ms=1500.0,
            queue_size=12,
            dropped_frames=3,
        )

        t = snap.telemetry
        assert t.input_fps == pytest.approx(30.0)
        assert t.processing_fps == pytest.approx(0.6)
        assert t.total_latency_ms == pytest.approx(1500.0)
        assert t.queue_size == 12
        assert t.dropped_frames == 3
        assert t.active_objects == 2
        assert t.active_relations == 1

    def test_camera_pose_conversion(self):
        """world_T_camera is converted to nested tuples."""
        graph = _build_test_graph()
        packet = _make_packet()
        snap = create_snapshot(graph, packet)

        assert snap.world_T_camera is not None
        assert len(snap.world_T_camera) == 4
        assert len(snap.world_T_camera[0]) == 4
        # Identity matrix check
        assert snap.world_T_camera[0][0] == pytest.approx(1.0)
        assert snap.world_T_camera[3][3] == pytest.approx(1.0)
        assert snap.world_T_camera[0][1] == pytest.approx(0.0)

    def test_empty_graph(self):
        """Snapshot from empty graph produces empty tuples, not errors."""
        graph = TemporalSceneGraph()
        graph.current_frame_index = 0
        graph.current_timestamp = 0.0
        packet = _make_packet(frame_index=0, timestamp=0.0)

        snap = create_snapshot(graph, packet)
        assert snap.objects == ()
        assert snap.relations == ()
        assert snap.telemetry.active_objects == 0
        assert snap.telemetry.active_relations == 0

    def test_no_numpy_in_snapshot(self):
        """Verify no numpy arrays leak into the snapshot."""
        graph = _build_test_graph()
        packet = _make_packet()
        snap = create_snapshot(graph, packet)

        for obj in snap.objects:
            for field_name in ("centroid_world", "velocity_world",
                               "obb_center_world", "obb_extents_world",
                               "bbox_min_world", "bbox_max_world"):
                val = getattr(obj, field_name)
                if val is not None:
                    assert isinstance(val, tuple), f"{field_name} is {type(val)}, expected tuple"
                    assert all(isinstance(x, float) for x in val)

            if obj.obb_axes_world is not None:
                assert isinstance(obj.obb_axes_world, tuple)
                for row in obj.obb_axes_world:
                    assert isinstance(row, tuple)
                    assert all(isinstance(x, float) for x in row)
