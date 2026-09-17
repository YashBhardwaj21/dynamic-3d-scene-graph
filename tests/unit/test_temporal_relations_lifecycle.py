"""Unit tests for Stage 11: Temporal Relations & Relation State Machine.

Validates:
1. Full relation lifecycle: PROPOSED -> ACTIVE -> DECAYING -> UNKNOWN -> TERMINATED
2. Temporal hysteresis via min_confirmation_hits
3. Smooth continuous belief decay
4. Contradiction transitions and thresholds
5. Relation evidence filtering (reflexive rejection, confidence thresholding, bounds checking)
6. Graph edge persistence during temporary occlusion/decay ("Unknown != False")
7. Query helpers and backward-compatible enum equivalence
"""

import numpy as np
import pytest

from scene_graph.config import RelationTemporalConfig
from scene_graph.geometry.reference_frame import ReferenceFrameType
from scene_graph.graph.edge import GraphEdge
from scene_graph.graph.node import GraphNode
from scene_graph.graph.participation_state import GraphParticipationState
from scene_graph.graph.temporal_graph import TemporalSceneGraph
from scene_graph.relations.evidence import EvidenceResult, RelationEvidence
from scene_graph.temporal.object_state import ObjectState
from scene_graph.temporal.relation_state import RelationKey, RelationState, RelationStateMachine
from scene_graph.tracking.track import Track, TrackState


def make_dummy_track(track_id: str, class_name: str = "cup") -> Track:
    return Track(
        object_id=track_id,
        class_name=class_name,
        state=TrackState.ACTIVE,
        _initial_centroid=np.array([0.0, 0.0, 0.5]),
        last_observed_frame=1,
        first_observed_frame=1,
        observation_count=3,
        missing_count=0,
        detection_confidence=0.9,
        track_observation_ratio=1.0,
        recent_observations=[],
        last_timestamp=1.0,
    )


def make_evidence(
    subject_id: str = "cup_1",
    object_id: str = "table_1",
    predicate: str = "ON",
    frame_index: int = 0,
    timestamp: float = 0.0,
    result: EvidenceResult = EvidenceResult.SUPPORTED,
    confidence: float = 0.9,
) -> RelationEvidence:
    return RelationEvidence(
        predicate=predicate,
        subject_id=subject_id,
        object_id=object_id,
        frame_index=frame_index,
        timestamp=timestamp,
        result=result,
        value=0.02,
        threshold=0.05,
        confidence=confidence,
        reference_frame=ReferenceFrameType.WORLD,
        evidence_type="support",
        details={},
    )


def test_relation_state_enum_equivalence_and_aliases():
    """Validates bidirectional equality between modern lifecycle states and legacy aliases."""
    # ACTIVE == SUPPORTED == CONFIRMED
    assert RelationState.ACTIVE == RelationState.SUPPORTED
    assert RelationState.SUPPORTED == RelationState.ACTIVE
    assert RelationState.ACTIVE == RelationState.CONFIRMED
    assert RelationState.ACTIVE == "active"
    assert RelationState.ACTIVE == "supported"
    assert RelationState.SUPPORTED == "active"

    # PROPOSED == HYPOTHESIZED
    assert RelationState.PROPOSED == RelationState.HYPOTHESIZED
    assert RelationState.PROPOSED == "proposed"
    assert RelationState.PROPOSED == "hypothesized"

    # DECAYING == WEAKENING
    assert RelationState.DECAYING == RelationState.WEAKENING
    assert RelationState.DECAYING == "decaying"
    assert RelationState.DECAYING == "weakening"

    # TERMINATED == ENDED
    assert RelationState.TERMINATED == RelationState.ENDED
    assert RelationState.TERMINATED == "terminated"
    assert RelationState.TERMINATED == "ended"

    # Parsing through _missing_
    assert RelationState("supported") == RelationState.ACTIVE
    assert RelationState("SUPPORTED") == RelationState.ACTIVE
    assert RelationState("proposed") == RelationState.PROPOSED
    assert RelationState("decaying") == RelationState.DECAYING
    assert RelationState("terminated") == RelationState.TERMINATED


def test_relation_lifecycle_proposed_to_active_with_hysteresis():
    """Validates temporal hysteresis: requires min_confirmation_hits consecutive hits before ACTIVE."""
    config = RelationTemporalConfig(
        confirmation_threshold=0.7,
        contradiction_threshold=-0.7,
        decay_per_second=0.1,
        unknown_after_seconds=2.0,
        lost_after_seconds=5.0,
        min_confirmation_hits=2,
    )
    sm = RelationStateMachine(config)
    key: RelationKey = ("cup_1", "table_1", "ON")

    # Frame 0: 1st hit. Belief rises to 0.8 >= 0.7, but hit_count is 1 < 2 -> PROPOSED
    ev0 = make_evidence(frame_index=0, timestamp=0.0, confidence=0.8)
    states0 = sm.update([ev0], frame_index=0, timestamp=0.0)
    assert states0[key] == RelationState.PROPOSED
    assert sm.get_hit_count(key) == 1
    assert not sm.is_confirmed(key)

    # Frame 1: 2nd consecutive hit. hit_count becomes 2 >= 2 -> transitions to ACTIVE / SUPPORTED
    ev1 = make_evidence(frame_index=1, timestamp=0.1, confidence=0.8)
    states1 = sm.update([ev1], frame_index=1, timestamp=0.1)
    assert states1[key] == RelationState.ACTIVE
    assert states1[key] == RelationState.SUPPORTED
    assert sm.get_hit_count(key) == 2
    assert sm.is_confirmed(key)


def test_relation_lifecycle_decay_to_unknown_to_terminated():
    """Validates transition from ACTIVE -> DECAYING -> UNKNOWN -> TERMINATED on observation loss."""
    config = RelationTemporalConfig(
        confirmation_threshold=0.6,
        contradiction_threshold=-0.6,
        decay_per_second=0.5,
        unknown_after_seconds=1.0,
        lost_after_seconds=2.0,
        min_confirmation_hits=1,
    )
    sm = RelationStateMachine(config)
    key: RelationKey = ("cup_1", "table_1", "ON")

    # Frame 0: Confirmed active
    ev = make_evidence(frame_index=0, timestamp=0.0, confidence=0.8)
    sm.update([ev], frame_index=0, timestamp=0.0)
    assert sm.get_state(key) == RelationState.ACTIVE
    assert sm.get_belief(key) == 0.8

    # Frame 1: 0.5s later, no observation.
    # missing_time = 0.5 < unknown_after_seconds (1.0).
    # belief decays by 0.5 * 0.5 = 0.25 -> belief = 0.55 < confirmation_threshold (0.6).
    # Since previously ACTIVE, transitions to DECAYING
    states1 = sm.update([], frame_index=1, timestamp=0.5)
    assert key in states1
    assert states1[key] == RelationState.DECAYING
    assert sm.get_miss_count(key) == 1
    assert pytest.approx(sm.get_belief(key), abs=1e-3) == 0.55

    # Frame 2: 1.2s later (missing_time = 1.2 >= unknown_after_seconds = 1.0)
    # Transitions to UNKNOWN
    states2 = sm.update([], frame_index=2, timestamp=1.2)
    assert key in states2
    assert states2[key] == RelationState.UNKNOWN

    # Frame 3: 2.5s later (missing_time = 2.5 >= lost_after_seconds = 2.0)
    # Transitions to TERMINATED and is purged
    states3 = sm.update([], frame_index=3, timestamp=2.5)
    assert key not in states3
    assert sm.get_state(key) == RelationState.UNKNOWN  # Not in sm.states anymore


def test_relation_contradiction_transition():
    """Validates transition to CONTRADICTED when contradictory evidence reaches threshold."""
    config = RelationTemporalConfig(
        confirmation_threshold=0.6,
        contradiction_threshold=-0.6,
        decay_per_second=0.0,  # Zero decay to test evidence only
        unknown_after_seconds=5.0,
        lost_after_seconds=10.0,
        min_confirmation_hits=1,
    )
    sm = RelationStateMachine(config)
    key: RelationKey = ("cup_1", "table_1", "ON")

    # Start with confirmed active relation
    ev_pos = make_evidence(frame_index=0, timestamp=0.0, confidence=0.8, result=EvidenceResult.SUPPORTED)
    sm.update([ev_pos], frame_index=0, timestamp=0.0)
    assert sm.get_state(key) == RelationState.ACTIVE

    # Apply moderate contradictory evidence (confidence 0.5)
    # Belief drops from 0.8 to 0.3 (> contradiction_threshold)
    ev_neg1 = make_evidence(frame_index=1, timestamp=0.1, confidence=0.5, result=EvidenceResult.CONTRADICTED)
    states1 = sm.update([ev_neg1], frame_index=1, timestamp=0.1)
    assert states1[key] == RelationState.DECAYING

    # Apply second contradictory evidence (confidence 1.0)
    # Belief drops to 0.3 - 1.0 = -0.7 <= contradiction_threshold (-0.6)
    ev_neg2 = make_evidence(frame_index=2, timestamp=0.2, confidence=1.0, result=EvidenceResult.CONTRADICTED)
    states2 = sm.update([ev_neg2], frame_index=2, timestamp=0.2)
    assert states2[key] == RelationState.CONTRADICTED
    assert sm.get_hit_count(key) == 0


def test_relation_evidence_filtering():
    """Validates evidence filtering for reflexivity, low confidence, bounds, and deduplication."""
    config = RelationTemporalConfig(
        confirmation_threshold=0.5,
        contradiction_threshold=-0.5,
        decay_per_second=0.1,
        unknown_after_seconds=2.0,
        lost_after_seconds=5.0,
        min_evidence_confidence=0.25,
    )
    sm = RelationStateMachine(config)

    # 1. Reflexive evidence (cup_1 -> cup_1) must be ignored
    ev_refl = make_evidence(subject_id="cup_1", object_id="cup_1", confidence=0.9)
    # 2. Low confidence evidence (< 0.25) must be filtered
    ev_low = make_evidence(subject_id="cup_1", object_id="table_1", confidence=0.10)

    states = sm.update([ev_refl, ev_low], frame_index=0, timestamp=0.0)
    assert ("cup_1", "cup_1", "ON") not in states
    assert ("cup_1", "table_1", "ON") not in states

    # 3. Invalid confidence (> 1.0, < 0.0, nan, inf) raises ValueError
    with pytest.raises(ValueError):
        ev_nan = make_evidence(confidence=float("nan"))
        sm.update([ev_nan], frame_index=1, timestamp=0.1)

    with pytest.raises(ValueError):
        ev_neg = make_evidence(confidence=-0.1)
        sm.update([ev_neg], frame_index=1, timestamp=0.1)

    # 4. Deduplication in same frame: picks higher confidence
    ev_cand1 = make_evidence(subject_id="cup_1", object_id="table_1", confidence=0.5)
    ev_cand2 = make_evidence(subject_id="cup_1", object_id="table_1", confidence=0.9)
    states_dedup = sm.update([ev_cand1, ev_cand2], frame_index=2, timestamp=0.2)
    assert sm.get_belief(("cup_1", "table_1", "ON")) == 0.9


def test_temporal_graph_edge_retention_during_decay():
    """Validates that TemporalSceneGraph retains edges during DECAYING and UNKNOWN ('Unknown != False')."""
    graph = TemporalSceneGraph()
    graph.current_frame_index = 0
    graph.current_timestamp = 0.0

    track_a = make_dummy_track("cup_1")
    track_b = make_dummy_track("table_1", class_name="table")

    graph.nodes["cup_1"] = GraphNode(object_id="cup_1", class_name="cup", track=track_a, state=ObjectState.STABLE)
    graph.nodes["table_1"] = GraphNode(object_id="table_1", class_name="table", track=track_b, state=ObjectState.STABLE)

    key = ("cup_1", "table_1", "ON")
    ev = make_evidence(subject_id="cup_1", object_id="table_1", confidence=0.9)

    # 1. Active edge creation
    graph.update_edges([ev], {key: RelationState.ACTIVE})
    assert key in graph.edges
    assert graph.edges[key].is_active
    assert len(graph.get_active_edges()) == 1

    # 2. Transition to DECAYING: edge is RETAINED, participation is TEMPORARILY_UNOBSERVED
    graph.current_frame_index = 1
    graph.current_timestamp = 0.5
    graph.update_edges([], {key: RelationState.DECAYING})
    assert key in graph.edges  # Must NOT be deleted!
    assert graph.edges[key].state == RelationState.DECAYING
    assert graph.edges[key].participation == GraphParticipationState.TEMPORARILY_UNOBSERVED
    assert not graph.edges[key].is_active  # Not active for navigation/manipulation
    assert len(graph.get_active_edges()) == 0

    # 3. Transition to UNKNOWN: edge is STILL RETAINED with UNKNOWN state
    graph.current_frame_index = 2
    graph.current_timestamp = 1.2
    graph.update_edges([], {key: RelationState.UNKNOWN})
    assert key in graph.edges  # Still retained in graph memory
    assert graph.edges[key].state == RelationState.UNKNOWN

    # 4. Transition to TERMINATED or CONTRADICTED: edge is now deleted
    graph.current_frame_index = 3
    graph.current_timestamp = 2.5
    graph.update_edges([], {key: RelationState.TERMINATED})
    assert key not in graph.edges  # Cleanly removed
