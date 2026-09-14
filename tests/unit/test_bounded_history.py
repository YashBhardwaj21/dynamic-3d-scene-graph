"""Tests for Phase 11: Bounded event history, disk streaming, and typed snapshot querying."""

import json
import tempfile
from collections import deque
import numpy as np
import pytest

from scene_graph.graph.event import GraphEvent, GraphEventType
from scene_graph.graph.graph_history import GraphHistory
from scene_graph.graph.temporal_graph import TemporalSceneGraph
from scene_graph.graph.node import GraphNode
from scene_graph.graph.edge import GraphEdge
from scene_graph.graph.participation_state import GraphParticipationState
from scene_graph.graph.snapshot import (
    create_snapshot,
    SceneGraphSnapshot,
    ObjectSnapshot,
    RelationSnapshot,
)
from scene_graph.tracking.track import Track, TrackState
from scene_graph.tracking.state import KalmanState
from scene_graph.temporal.object_state import ObjectState
from scene_graph.temporal.relation_state import RelationState
from scene_graph.relations.evidence import RelationEvidence, EvidenceResult, ReferenceFrameType
from scene_graph.data.frame_packet import FramePacket
from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.geometry.reference_frame import RelationReferenceFrame


def _make_test_track(track_id, class_name, centroid, label_belief=None):
    centroid = np.asarray(centroid, dtype=np.float64)
    return Track(
        object_id=track_id,
        class_name=class_name,
        state=TrackState.ACTIVE,
        _initial_centroid=centroid.copy(),
        last_observed_frame=5,
        first_observed_frame=1,
        observation_count=5,
        missing_count=0,
        detection_confidence=0.88,
        track_observation_ratio=1.0,
        last_timestamp=1.0,
        recent_observations=deque([], maxlen=10),
        kalman_state=KalmanState(centroid.copy(), initial_cov_pos=1.0, initial_cov_vel=1.0),
        label_belief=label_belief or {class_name: 1.0},
    )


def test_bounded_history_length_cap():
    """Verify that GraphHistory strictly caps event count at maxlen over large iteration counts."""
    history = GraphHistory(maxlen=200)

    for i in range(5000):
        event = GraphEvent(
            event_type=GraphEventType.NODE_ADDED,
            timestamp=float(i) * 0.1,
            frame_index=i,
            subject_id=f"obj_{i % 10}",
        )
        history.add_event(event)

    assert len(history) == 200
    assert len(history.events) == 200
    # Oldest retained event should be frame 4800, latest frame 4999
    assert history.events[0].frame_index == 4800
    assert history.events[-1].frame_index == 4999


def test_bounded_history_disk_streaming():
    """Verify optional streaming to disk while keeping memory bounded."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as tmp:
        dump_path = tmp.name

    history = GraphHistory(maxlen=5, dump_file=dump_path)
    try:
        for i in range(25):
            history.add_event(GraphEvent(
                event_type=GraphEventType.EDGE_ADDED,
                timestamp=float(i),
                frame_index=i,
                subject_id="a",
                object_id="b",
                predicate="ON",
            ))

        assert len(history) == 5
        history.close()

        # Read back jsonl from disk
        with open(dump_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == 25
        first = json.loads(lines[0])
        last = json.loads(lines[-1])
        assert first["frame_index"] == 0
        assert first["predicate"] == "ON"
        assert last["frame_index"] == 24
    finally:
        import os
        if os.path.exists(dump_path):
            os.remove(dump_path)


def test_snapshot_label_belief_and_query():
    """Verify snapshot exports label_belief and provides programmatic query methods."""
    graph = TemporalSceneGraph(history_maxlen=100)
    graph.current_frame_index = 5
    graph.current_timestamp = 1.0

    track = _make_test_track(
        "obj_1", "mug", [0.0, 1.0, 0.5], label_belief={"mug": 0.75, "cup": 0.25}
    )
    graph.nodes["obj_1"] = GraphNode(
        object_id="obj_1",
        class_name="mug",
        track=track,
        state=ObjectState.STABLE,
    )

    track_table = _make_test_track(
        "obj_2", "table", [0.0, 1.0, 0.0], label_belief={"table": 0.99}
    )
    graph.nodes["obj_2"] = GraphNode(
        object_id="obj_2",
        class_name="table",
        track=track_table,
        state=ObjectState.STABLE,
    )

    evidence = RelationEvidence(
        predicate="ON",
        subject_id="obj_1",
        object_id="obj_2",
        frame_index=5,
        timestamp=1.0,
        result=EvidenceResult.SUPPORTED,
        value=0.02,
        threshold=0.08,
        confidence=0.92,
        reference_frame=ReferenceFrameType.WORLD,
        evidence_type="support",
        details={},
    )
    graph.edges[("obj_1", "obj_2", "ON")] = GraphEdge(
        predicate="ON",
        subject_id="obj_1",
        object_id="obj_2",
        state=RelationState.SUPPORTED,
        latest_evidence=evidence,
        participation=GraphParticipationState.ACTIVE,
    )

    packet = FramePacket(
        frame_index=5,
        timestamp=1.0,
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        depth=np.ones((10, 10), dtype=np.float32),
        world_T_camera=np.eye(4),
        camera_intrinsics=CameraIntrinsics(fx=500, fy=500, cx=5, cy=5, width=10, height=10),
        depth_model=DepthModel(scale=1000.0),
    )

    snap = create_snapshot(graph, packet)

    # Check label_belief field on ObjectSnapshot
    obj1 = [o for o in snap.objects if o.track_id == "obj_1"][0]
    assert obj1.label_belief == {"mug": 0.75, "cup": 0.25}

    # Query tests
    mugs = snap.query_objects(class_name="mug")
    assert len(mugs) == 1
    assert mugs[0].track_id == "obj_1"

    nonexistent = snap.query_objects(class_name="chair")
    assert len(nonexistent) == 0

    on_relations = snap.query_relations(predicate="ON")
    assert len(on_relations) == 1
    assert on_relations[0].subject_id == "obj_1"
    assert on_relations[0].object_id == "obj_2"

    # Verify serialization to dict
    d = snap.to_dict()
    assert d["frame_index"] == 5
    assert len(d["objects"]) == 2
    assert d["objects"][0]["label_belief"] == {"mug": 0.75, "cup": 0.25}
    # Must be JSON serializable
    json_str = json.dumps(d)
    assert "mug" in json_str
