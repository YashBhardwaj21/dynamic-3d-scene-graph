"""Unit tests for Stage 6 Goal Resolution and Prompt Parsing."""

from pathlib import Path
import pytest

from scene_graph.navigation.goal_resolver import GoalResolver


def test_goal_resolver_target_vs_anchor_role():
    """An explicit ID in the anchor role must NOT override the grammatical target."""
    sem_map = {
        "entities": {
            "desk_003": {
                "class_name": "desk",
                "is_confirmed": True,
                "centroid_floor": [2.0, 0.0, 0.4],
                "bbox_min_floor": [1.5, -0.4, 0.0],
                "bbox_max_floor": [2.5, 0.4, 0.8],
            },
            "chair_001": {
                "class_name": "chair",
                "is_confirmed": True,
                "centroid_floor": [2.3, 0.5, 0.4],  # ~0.5m from desk_003
                "bbox_min_floor": [2.1, 0.3, 0.0],
                "bbox_max_floor": [2.5, 0.7, 0.8],
            },
            "chair_002": {
                "class_name": "chair",
                "is_confirmed": True,
                "centroid_floor": [8.0, 8.0, 0.4],  # Far from desk_003
                "bbox_min_floor": [7.8, 7.8, 0.0],
                "bbox_max_floor": [8.2, 8.2, 0.8],
            },
        }
    }

    resolver = GoalResolver(sem_map)

    # Prompt: "go to the chair near desk_003 and stop 0.5 metres away"
    res = resolver.resolve("go to the chair near desk_003 and stop 0.5 metres away")
    assert res.success is True
    # The target MUST be chair_001, NOT desk_003!
    assert res.target_entity_id == "chair_001"
    assert res.target_class == "chair"
    assert res.approach_pose_2d is not None


def test_goal_resolver_rejects_ambiguity_no_silent_guessing():
    """If multiple candidates match without distinguishing relation, reject ambiguity."""
    sem_map = {
        "entities": {
            "chair_001": {
                "class_name": "chair",
                "is_confirmed": True,
                "centroid_floor": [1.0, 1.0, 0.4],
            },
            "chair_002": {
                "class_name": "chair",
                "is_confirmed": True,
                "centroid_floor": [2.0, 2.0, 0.4],
            },
        }
    }

    resolver = GoalResolver(sem_map)
    # Prompt with ambiguous target
    res = resolver.resolve("go to the chair and stop 0.5 metres away")
    assert res.success is False
    assert "Ambiguous goal" in res.rejection_reason
    assert "chair_001" in res.rejection_reason
    assert "chair_002" in res.rejection_reason


def test_goal_resolver_explicit_target_id():
    """Explicit target ID resolves directly."""
    sem_map = {
        "entities": {
            "desk_003": {
                "class_name": "desk",
                "is_confirmed": True,
                "centroid_floor": [2.0, 0.0, 0.4],
            },
        }
    }
    resolver = GoalResolver(sem_map)
    res = resolver.resolve("go to desk_003 and stop 0.4 metres away")
    assert res.success is True
    assert res.target_entity_id == "desk_003"
    assert res.standoff_distance_m == 0.4
