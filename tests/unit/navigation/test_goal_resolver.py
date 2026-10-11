"""Unit tests for Natural Language Prompt to Navigation Goal Resolver."""

import math
import numpy as np
import pytest

from scene_graph.navigation.goal_resolver import GoalResolver, GoalResolutionResult


@pytest.fixture
def sample_semantic_map():
    return {
        "metadata": {"map_name": "test_room", "frame_id": "floor_map"},
        "entities": {
            "desk_001": {
                "class_name": "desk",
                "status": "CONFIRMED",
                "floor_coordinates": {
                    "center": [2.0, 1.0, 0.75],
                    "bounding_box_min_max": {"min_x": 1.5, "max_x": 2.5, "min_y": 0.6, "max_y": 1.4},
                },
                "dimensions_m": {"width_x": 1.0, "length_y": 0.8, "height_z": 0.75},
            },
            "desk_002": {
                "class_name": "desk",
                "status": "CONFIRMED",
                "floor_coordinates": {
                    "center": [-1.0, 3.0, 0.75],
                    "bounding_box_min_max": {"min_x": -1.5, "max_x": -0.5, "min_y": 2.6, "max_y": 3.4},
                },
                "dimensions_m": {"width_x": 1.0, "length_y": 0.8, "height_z": 0.75},
            },
            "robot_drone_001": {
                "class_name": "robot",
                "status": "CONFIRMED",
                "floor_coordinates": {
                    "center": [2.1, 1.1, 0.85],  # On top of desk_001
                    "bounding_box_min_max": {"min_x": 1.95, "max_x": 2.25, "min_y": 0.95, "max_y": 1.25},
                },
                "dimensions_m": {"width_x": 0.3, "length_y": 0.3, "height_z": 0.2},
            },
            "chair_003": {
                "class_name": "chair",
                "status": "CONFIRMED",
                "floor_coordinates": {
                    "center": [0.5, -0.5, 0.45],
                    "bounding_box_min_max": {"min_x": 0.25, "max_x": 0.75, "min_y": -0.75, "max_y": -0.25},
                },
                "dimensions_m": {"width_x": 0.5, "length_y": 0.5, "height_z": 0.9},
            },
            "projector_001": {
                "class_name": "projector",
                "status": "QUARANTINED",
                "floor_coordinates": {
                    "center": [0.0, 2.0, 2.2],
                },
            },
        },
    }


def test_resolve_explicit_entity_id(sample_semantic_map):
    resolver = GoalResolver(sample_semantic_map)
    res = resolver.resolve("Go to chair_003 and stop 0.4 metres away.", current_robot_pose=(0.0, 0.0, 0.0))

    assert res.success is True
    assert res.target_entity_id == "chair_003"
    assert res.standoff_distance_m == 0.4
    assert res.approach_pose_2d is not None

    gx, gy, gyaw = res.approach_pose_2d
    # Goal must be outside the chair
    cx, cy, _ = res.target_center_floor
    dist_to_center = math.hypot(gx - cx, gy - cy)
    assert dist_to_center > 0.4 + 0.18

    # Yaw must point toward object
    expected_yaw = math.atan2(cy - gy, cx - gx)
    assert abs(gyaw - expected_yaw) < 1e-4


def test_resolve_relational_target(sample_semantic_map):
    """'table with the drone' must resolve desk_001 (which has drone on it), not desk_002."""
    resolver = GoalResolver(sample_semantic_map)
    res = resolver.resolve("Go to the table with the drone and stop 0.5 metres away", current_robot_pose=(0.0, 0.0, 0.0))

    assert res.success is True
    assert res.target_entity_id == "desk_001"
    assert res.standoff_distance_m == 0.5
    assert res.evidence["match_type"] in ("relational_association", "closest_to_relational_cue")
    assert res.evidence["related_entity"] == "robot_drone_001"


def test_reject_unconfirmed_entity(sample_semantic_map):
    resolver = GoalResolver(sample_semantic_map)
    res = resolver.resolve("Go to projector_001", current_robot_pose=(0.0, 0.0, 0.0))

    assert res.success is False
    assert "quarantined or unconfirmed" in res.rejection_reason


def test_reject_missing_entity(sample_semantic_map):
    resolver = GoalResolver(sample_semantic_map)
    res = resolver.resolve("Go to refrigerator_999", current_robot_pose=(0.0, 0.0, 0.0))

    assert res.success is False
    assert "No confirmed entities" in res.rejection_reason
