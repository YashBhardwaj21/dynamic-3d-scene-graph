import numpy as np
import pytest

from scene_graph.ontology import (
    EntityType,
    EntityLifecycleState,
    VisibilityState,
    SemanticHypothesis,
    PersistentEntity,
    RelationCategory,
    RelationState,
    RelationPredicate,
    CANONICAL_PREDICATES,
    PREDICATE_CATEGORIES,
    INVERSE_PREDICATES,
    SYMMETRIC_PREDICATES,
    RelationEdge,
    normalize_predicate,
    get_inverse_predicate,
    is_canonical_predicate,
    get_predicate_category,
)
from scene_graph.graph.node import GraphNode
from scene_graph.graph.edge import GraphEdge
from scene_graph.tracking.track import Track, TrackState
from scene_graph.temporal.object_state import ObjectState
from scene_graph.temporal.relation_state import RelationState as LegacyRelationState
from scene_graph.relations.evidence import RelationEvidence, EvidenceResult, ReferenceFrameType


def test_entity_types():
    assert set(e.value for e in EntityType) == {
        "WORLD", "PLACE", "SURFACE", "CONTAINER", "OBJECT", "AGENT", "ROBOT"
    }


def test_entity_lifecycle_states():
    assert set(s.value for s in EntityLifecycleState) == {
        "TENTATIVE", "CONFIRMED", "VISIBLE", "OCCLUDED", "STALE", "LOST", "ARCHIVED"
    }


def test_visibility_states():
    assert set(v.value for v in VisibilityState) == {
        "VISIBLE", "OCCLUDED", "OUT_OF_VIEW", "UNKNOWN"
    }


def test_semantic_hypothesis():
    hypo = SemanticHypothesis(label="mug", confidence=0.88, source="yoloe")
    d = hypo.to_dict()
    assert d["label"] == "mug"
    assert d["confidence"] == 0.88
    assert d["source"] == "yoloe"

    restored = SemanticHypothesis.from_dict(d)
    assert restored.label == "mug"
    assert restored.confidence == 0.88


def test_persistent_entity_schema():
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = 1.2
    pose[1, 3] = -0.5
    pose[2, 3] = 0.85

    entity = PersistentEntity(
        entity_id="entity_000042",
        entity_type=EntityType.OBJECT,
        semantic_hypotheses=[
            SemanticHypothesis(label="cup", confidence=0.6),
            SemanticHypothesis(label="mug", confidence=0.92),
        ],
        pose=pose,
        velocity=np.array([0.0, 0.0, 0.0]),
        uncertainty=np.eye(3) * 0.05,
        first_seen=100.0,
        last_seen=105.0,
        last_observed=104.5,
        visibility_state=VisibilityState.VISIBLE,
        lifecycle_state=EntityLifecycleState.CONFIRMED,
        source_track_ids=["track_17"],
        parent_context_id="desk_0004",
        attributes={"color": "white"},
    )

    assert entity.primary_label == "mug"
    assert entity.primary_confidence == 0.92
    assert entity.is_active is True

    d = entity.to_dict()
    assert d["entity_id"] == "entity_000042"
    assert d["entity_type"] == "OBJECT"
    assert d["parent_context_id"] == "desk_0004"

    reconstructed = PersistentEntity.from_dict(d)
    assert reconstructed.entity_id == entity.entity_id
    assert reconstructed.primary_label == "mug"
    assert reconstructed.pose.shape == (4, 4)
    np.testing.assert_allclose(reconstructed.pose, pose)


def test_canonical_predicates_and_categories():
    assert len(CANONICAL_PREDICATES) == 12

    structural = {
        RelationPredicate.SUPPORTED_BY.value,
        RelationPredicate.INSIDE.value,
        RelationPredicate.ATTACHED_TO.value,
    }
    spatial = {
        RelationPredicate.NEAR.value,
        RelationPredicate.TOUCHING.value,
        RelationPredicate.LEFT_OF.value,
        RelationPredicate.RIGHT_OF.value,
        RelationPredicate.ABOVE.value,
        RelationPredicate.BELOW.value,
        RelationPredicate.FRONT_OF.value,
        RelationPredicate.BEHIND.value,
    }
    visibility = {
        RelationPredicate.OCCLUDES.value,
    }

    assert structural | spatial | visibility == CANONICAL_PREDICATES

    for p in structural:
        assert get_predicate_category(p) == RelationCategory.STRUCTURAL
    for p in spatial:
        assert get_predicate_category(p) == RelationCategory.SPATIAL
    for p in visibility:
        assert get_predicate_category(p) == RelationCategory.VISIBILITY


def test_canonical_inverse_algebra():
    expected_inverses = {
        "SUPPORTED_BY": "SUPPORTS",
        "SUPPORTS": "SUPPORTED_BY",
        "INSIDE": "CONTAINS",
        "CONTAINS": "INSIDE",
        "ATTACHED_TO": "ATTACHED_TO",
        "NEAR": "NEAR",
        "TOUCHING": "TOUCHING",
        "LEFT_OF": "RIGHT_OF",
        "RIGHT_OF": "LEFT_OF",
        "ABOVE": "BELOW",
        "BELOW": "ABOVE",
        "FRONT_OF": "BEHIND",
        "BEHIND": "FRONT_OF",
        "OCCLUDES": "OCCLUDED_BY",
        "OCCLUDED_BY": "OCCLUDES",
    }

    for pred, inv in expected_inverses.items():
        assert get_inverse_predicate(pred) == inv
        # Double inverse symmetry
        assert get_inverse_predicate(get_inverse_predicate(pred)) == pred


def test_legacy_predicate_normalization():
    assert normalize_predicate("ON") == "SUPPORTED_BY"
    assert normalize_predicate("UNDER") == "SUPPORTS"
    assert normalize_predicate("CONTAINING") == "CONTAINS"
    assert normalize_predicate("IN_FRONT_OF") == "FRONT_OF"
    assert normalize_predicate("OCCLUDING") == "OCCLUDES"
    assert normalize_predicate("SUPPORTED_BY") == "SUPPORTED_BY"
    assert normalize_predicate("NEAR") == "NEAR"

    # Inverse lookup on legacy aliases works seamlessly
    assert get_inverse_predicate("ON") == "SUPPORTS"
    assert get_inverse_predicate("UNDER") == "SUPPORTED_BY"
    assert get_inverse_predicate("IN_FRONT_OF") == "BEHIND"
    assert get_inverse_predicate("OCCLUDING") == "OCCLUDED_BY"


def test_relation_edge_schema():
    edge = RelationEdge(
        relation_id="rel_001",
        subject_entity_id="entity_001",
        predicate="SUPPORTED_BY",
        object_entity_id="surface_001",
        state=RelationState.CONFIRMED,
        confidence=0.95,
        uncertainty=0.05,
        first_confirmed=10.0,
        last_confirmed=12.5,
        source_estimators=["support_estimator"],
        evidence_summary={"contact_distance_m": 0.012, "overlap_ratio": 0.85},
        frame_id=45,
        world_timestamp=12.5,
    )

    assert edge.canonical_key == ("entity_001", "surface_001", "SUPPORTED_BY")
    assert edge.is_active is True

    d = edge.to_dict()
    assert d["predicate"] == "SUPPORTED_BY"
    assert d["state"] == "CONFIRMED"

    reconstructed = RelationEdge.from_dict(d)
    assert reconstructed.predicate == "SUPPORTED_BY"
    assert reconstructed.confidence == 0.95


def test_graph_node_and_edge_ontology_bridges():
    from collections import deque
    track = Track(
        object_id="track_001",
        class_name="mug",
        state=TrackState.ACTIVE,
        _initial_centroid=np.array([0.5, 0.2, 0.8]),
        last_observed_frame=1,
        first_observed_frame=1,
        observation_count=5,
        missing_count=0,
        detection_confidence=0.91,
        track_observation_ratio=1.0,
        last_timestamp=2.0,
        recent_observations=deque(),
    )
    node = GraphNode(
        object_id="track_001",
        class_name="mug",
        track=track,
        state=ObjectState.STABLE,
        spatial_context_id="surface_desk",
    )

    entity = node.to_persistent_entity()
    assert entity.entity_id == "track_001"
    assert entity.primary_label == "mug"
    assert entity.parent_context_id == "surface_desk"

    ev = RelationEvidence(
        predicate="ON",
        subject_id="track_001",
        object_id="surface_desk",
        frame_index=10,
        timestamp=2.0,
        result=EvidenceResult.SUPPORTED,
        value=0.015,
        threshold=0.08,
        confidence=0.91,
        reference_frame=ReferenceFrameType.WORLD,
        evidence_type="support_plane_contact",
        details={"contact_distance_m": 0.015},
    )
    edge = GraphEdge(
        predicate="ON",
        subject_id="track_001",
        object_id="surface_desk",
        state=LegacyRelationState.SUPPORTED,
        latest_evidence=ev,
        start_time=1.0,
        start_frame=5,
    )

    assert edge.canonical_predicate == "SUPPORTED_BY"
    onto_edge = edge.to_relation_edge()
    assert onto_edge.predicate == "SUPPORTED_BY"
    assert onto_edge.state == RelationState.CONFIRMED
    assert onto_edge.confidence == 0.91
