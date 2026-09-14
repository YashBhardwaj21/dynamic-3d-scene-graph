import numpy as np
import pytest

from scene_graph.config import RelationTemporalConfig
from scene_graph.geometry.reference_frame import ReferenceFrameType
from scene_graph.relations.evidence import EvidenceResult, RelationEvidence
from scene_graph.temporal.relation_state import RelationState, RelationStateMachine


def test_evidence_strength_bounds_and_finite_validation():
    """Asserts invalid evidence strength values (<0, >1, nan, inf) raise ValueError."""
    # Valid evidence strength
    ev = RelationEvidence(
        predicate="ON",
        subject_id="A",
        object_id="B",
        frame_index=0,
        timestamp=0.0,
        result=EvidenceResult.SUPPORTED,
        value=0.02,
        threshold=0.08,
        confidence=0.85,
        reference_frame=ReferenceFrameType.WORLD,
        evidence_type="support",
        details={},
    )
    assert ev.evidence_strength == 0.85

    # Negative confidence
    with pytest.raises(ValueError):
        RelationEvidence(
            predicate="ON",
            subject_id="A",
            object_id="B",
            frame_index=0,
            timestamp=0.0,
            result=EvidenceResult.SUPPORTED,
            value=0.02,
            threshold=0.08,
            confidence=-0.1,
            reference_frame=ReferenceFrameType.WORLD,
            evidence_type="support",
            details={},
        )

    # Exceeding 1.0
    with pytest.raises(ValueError):
        RelationEvidence(
            predicate="ON",
            subject_id="A",
            object_id="B",
            frame_index=0,
            timestamp=0.0,
            result=EvidenceResult.SUPPORTED,
            value=0.02,
            threshold=0.08,
            confidence=1.05,
            reference_frame=ReferenceFrameType.WORLD,
            evidence_type="support",
            details={},
        )

    # NaN
    with pytest.raises(ValueError):
        RelationEvidence(
            predicate="ON",
            subject_id="A",
            object_id="B",
            frame_index=0,
            timestamp=0.0,
            result=EvidenceResult.SUPPORTED,
            value=0.02,
            threshold=0.08,
            confidence=float("nan"),
            reference_frame=ReferenceFrameType.WORLD,
            evidence_type="support",
            details={},
        )

    # Inf
    with pytest.raises(ValueError):
        RelationEvidence(
            predicate="ON",
            subject_id="A",
            object_id="B",
            frame_index=0,
            timestamp=0.0,
            result=EvidenceResult.SUPPORTED,
            value=0.02,
            threshold=0.08,
            confidence=float("inf"),
            reference_frame=ReferenceFrameType.WORLD,
            evidence_type="support",
            details={},
        )


def test_monotonic_belief_drop_on_contradiction():
    """Asserts repeated contradictory evidence drops belief monotonically from +1.0 toward -1.0."""
    config = RelationTemporalConfig(
        confirmation_threshold=0.8,
        contradiction_threshold=-0.8,
        decay_per_second=0.0,  # Zero decay to isolate evidence impact
        unknown_after_seconds=10.0,
        lost_after_seconds=20.0,
    )
    machine = RelationStateMachine(config)
    key = ("A", "B", "ON")

    # Positive evidence pushes belief to +1.0
    ev_pos = RelationEvidence(
        predicate="ON",
        subject_id="A",
        object_id="B",
        frame_index=0,
        timestamp=0.0,
        result=EvidenceResult.SUPPORTED,
        value=0.02,
        threshold=0.08,
        confidence=1.0,
        reference_frame=ReferenceFrameType.WORLD,
        evidence_type="support",
        details={},
    )
    states = machine.update([ev_pos], frame_index=0, timestamp=0.0)
    assert states[key] == RelationState.SUPPORTED
    assert machine.beliefs[key] == 1.0

    # Sequential contradictory evidence (confidence 0.5 each)
    ev_neg = RelationEvidence(
        predicate="ON",
        subject_id="A",
        object_id="B",
        frame_index=1,
        timestamp=0.1,
        result=EvidenceResult.CONTRADICTED,
        value=0.5,
        threshold=0.08,
        confidence=0.5,
        reference_frame=ReferenceFrameType.WORLD,
        evidence_type="support",
        details={},
    )

    prev_belief = machine.beliefs[key]
    for frame_idx in range(1, 6):
        machine.update([ev_neg], frame_index=frame_idx, timestamp=float(frame_idx) * 0.1)
        curr_belief = machine.beliefs[key]
        assert curr_belief <= prev_belief
        prev_belief = curr_belief

    # Clamped at -1.0
    assert machine.beliefs[key] == -1.0
    assert machine.states[key] == RelationState.CONTRADICTED
