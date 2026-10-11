"""Unit tests for Stage 5 Object Association, Class Gating, and Canonicalization."""

import numpy as np
import pytest

from scene_graph.mapping.global_associator import (
    GlobalObjectAssociator,
    ObjectObservationRecord,
)


def test_associator_never_matches_incompatible_classes():
    """Observations of incompatible classes must NEVER associate, regardless of proximity."""
    associator = GlobalObjectAssociator(association_dist_threshold_m=0.80)

    # Observation 1: chair at (1.0, 1.0, 0.0)
    obs1 = ObjectObservationRecord(
        obs_id="obs_001",
        frame_index=0,
        timestamp=0.0,
        class_name="chair",
        confidence=0.9,
        bbox_xyxy=[10, 10, 50, 50],
        centroid_camera=[0.0, 0.0, 1.0],
        centroid_floor=[1.0, 1.0, 0.4],
        points_floor=np.array([[1.0, 1.0, 0.4]]),
    )
    assigned1 = associator.update([obs1])
    chair_id = assigned1["obs_001"]
    assert "chair" in chair_id

    # Observation 2: desk at virtually the EXACT SAME location (1.05, 1.05, 0.4) - 7cm away!
    obs2 = ObjectObservationRecord(
        obs_id="obs_002",
        frame_index=1,
        timestamp=0.1,
        class_name="desk",  # INCOMPATIBLE CLASS!
        confidence=0.85,
        bbox_xyxy=[10, 10, 50, 50],
        centroid_camera=[0.0, 0.0, 1.0],
        centroid_floor=[1.05, 1.05, 0.4],
        points_floor=np.array([[1.05, 1.05, 0.4]]),
    )
    assigned2 = associator.update([obs2])
    desk_id = assigned2["obs_002"]

    # Must be two distinct entities! Desk must NEVER match or contaminate chair!
    assert desk_id != chair_id
    assert "desk" in desk_id


def test_associator_merges_canonicalize_all_observation_records():
    """When entities are merged, all observations must have their global_entity_id updated."""
    associator = GlobalObjectAssociator(association_dist_threshold_m=0.80, min_hits_to_confirm=1)

    # Create two chair fragments initially separated (>0.80m) so they form distinct entities
    obs1 = ObjectObservationRecord(
        obs_id="obs_01", frame_index=0, timestamp=0.0, class_name="chair", confidence=0.9,
        bbox_xyxy=[0,0,10,10], centroid_camera=[0,0,1], centroid_floor=[2.0, 2.0, 0.4],
        points_floor=np.random.normal(loc=[2.0, 2.0, 0.4], scale=0.1, size=(50, 3)),
    )
    associator.update([obs1])

    obs2 = ObjectObservationRecord(
        obs_id="obs_02", frame_index=1, timestamp=0.1, class_name="chair", confidence=0.85,
        bbox_xyxy=[0,0,10,10], centroid_camera=[0,0,1], centroid_floor=[3.2, 3.2, 0.4],
        points_floor=np.random.normal(loc=[3.2, 3.2, 0.4], scale=0.1, size=(50, 3)),
    )
    associator.update([obs2])

    assert len(associator.unconfirmed_entities) + len(associator.entities) == 2

    # Promote to confirmed
    associator.evaluate_confirmation_policy()
    assert len(associator.entities) == 2

    # Now simulate refined spatial proximity: chair_002 is within 0.25m of chair_001
    associator.entities["chair_002"].centroid_floor = np.array([2.15, 2.15, 0.4])
    associator.merge_coincident_entities(merge_distance_threshold_m=0.40)

    # After merge, exactly 1 entity remains
    remaining_entities = list(associator.entities.values()) + list(associator.unconfirmed_entities.values())
    assert len(remaining_entities) == 1
    canonical_ent = remaining_entities[0]

    # Crucial check: EVERY observation in canonical_ent must have global_entity_id == canonical_ent.entity_id!
    for o in canonical_ent.observations:
        assert o.global_entity_id == canonical_ent.entity_id
