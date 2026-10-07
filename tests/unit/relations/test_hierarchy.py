import numpy as np
import pytest
from collections import deque

from scene_graph.config import SpatialHierarchyConfig
from scene_graph.tracking.track import Track, TrackState
from scene_graph.relations.context import FrameContext, ObservationGeometry
from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.reference_frame import RelationReferenceFrame, CameraFrame
from scene_graph.relations.hierarchy import SpatialContext, SpatialContextManager


def make_track(obj_id: str, label: str, centroid: np.ndarray, obs_count: int = 10) -> Track:
    return Track(
        object_id=obj_id,
        class_name=label,
        state=TrackState.ACTIVE,
        _initial_centroid=centroid,
        last_observed_frame=1,
        first_observed_frame=1,
        observation_count=obs_count,
        missing_count=0,
        detection_confidence=0.9,
        track_observation_ratio=1.0,
        last_timestamp=1.0,
        recent_observations=deque(),
    )


def make_geometry(obj_id: str, bbox_min: np.ndarray, bbox_max: np.ndarray, points: np.ndarray = None) -> ObservationGeometry:
    centroid = (bbox_min + bbox_max) / 2.0
    if points is None:
        # Generate simple horizontal planar grid
        xs = np.linspace(bbox_min[0], bbox_max[0], 10)
        ys = np.linspace(bbox_min[1], bbox_max[1], 10)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_z = np.full_like(grid_x, bbox_max[2])
        points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)

    return ObservationGeometry(
        obs_id=f"obs_{obj_id}",
        track_id=obj_id,
        centroid_world=centroid,
        bbox_min_world=bbox_min,
        bbox_max_world=bbox_max,
        depth_stats=None,
        points_world_sampled=points,
        points_world=points,
        points_camera=None,
        mask=None,
        valid_point_count=len(points),
    )


def test_geometry_first_anchor_scoring():
    """Verify that anchor score is purely geometric and physical with zero class-name gating."""
    mgr = SpatialContextManager(SpatialHierarchyConfig())

    # Create a large horizontal surface labeled as 'unknown_object_42' (demonstrating zero class dependency)
    desk_track = make_track("track_01", "unknown_object_42", np.array([0.0, 1.0, 0.0]))
    desk_geom = make_geometry("track_01", np.array([-0.6, 0.6, -0.1]), np.array([0.6, 1.4, 0.1]))

    # Create small objects on top of the surface
    cup_track = make_track("track_02", "cup", np.array([0.1, 0.9, 0.2]))
    cup_geom = make_geometry("track_02", np.array([0.05, 0.85, 0.15]), np.array([0.15, 0.95, 0.25]))

    keyboard_track = make_track("track_03", "keyboard", np.array([-0.2, 0.9, 0.15]))
    keyboard_geom = make_geometry("track_03", np.array([-0.35, 0.8, 0.12]), np.array([-0.05, 1.0, 0.18]))

    all_tracks = [desk_track, cup_track, keyboard_track]
    geoms = {
        "track_01": desk_geom,
        "track_02": cup_geom,
        "track_03": keyboard_geom,
    }

    # Compute anchor score for the large surface
    score_surface, details_surface = mgr.compute_anchor_score(
        desk_track, desk_geom, all_tracks, geoms, up_axis=np.array([0.0, 0.0, 1.0])
    )

    # Compute anchor score for the cup
    score_cup, details_cup = mgr.compute_anchor_score(
        cup_track, cup_geom, all_tracks, geoms, up_axis=np.array([0.0, 0.0, 1.0])
    )

    assert score_surface >= 0.40, f"Surface should qualify as anchor, got {score_surface}"
    assert score_cup < 0.25, f"Cup should not qualify as anchor, got {score_cup}"
    assert details_surface["supported_objects_count"] >= 2


def test_extensible_context_hierarchy_and_membership():
    """Verify that contexts form a multi-level hierarchy and objects are properly grouped."""
    mgr = SpatialContextManager(SpatialHierarchyConfig())

    desk_track = make_track("desk_01", "desk", np.array([0.0, 1.0, 0.0]))
    desk_geom = make_geometry("desk_01", np.array([-0.8, 0.5, -0.2]), np.array([0.8, 1.5, 0.0]))

    keyboard_track = make_track("kb_01", "keyboard", np.array([0.0, 1.0, 0.05]))
    keyboard_geom = make_geometry("kb_01", np.array([-0.2, 0.9, 0.02]), np.array([0.2, 1.1, 0.08]))

    # An isolated object far away on the floor / separate
    lamp_track = make_track("lamp_01", "floor_lamp", np.array([3.0, 3.0, 0.0]))
    lamp_geom = make_geometry("lamp_01", np.array([2.9, 2.9, 0.0]), np.array([3.1, 3.1, 1.5]))

    all_tracks = [desk_track, keyboard_track, lamp_track]
    geoms = {
        "desk_01": desk_geom,
        "kb_01": keyboard_geom,
        "lamp_01": lamp_geom,
    }

    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    frame_ctx = FrameContext(
        frame_index=1,
        timestamp=1.0,
        intrinsics=intrinsics,
        world_T_camera=np.eye(4),
        reference_frame=RelationReferenceFrame.create("map", np.eye(4)),
        camera_frame=CameraFrame.from_camera_pose(np.eye(4)),
        depth_image=None,
        observation_geometry=geoms,
    )

    contexts = mgr.discover_contexts(all_tracks, frame_ctx)

    assert "world" in contexts
    desk_cid = "context_desk_01"
    assert desk_cid in contexts
    assert contexts[desk_cid].level == 1
    assert contexts[desk_cid].parent_context_id == "world"

    membership = mgr.assign_context_membership(all_tracks, contexts, frame_ctx)

    # Keyboard sits on desk -> belongs to desk context
    assert membership["kb_01"] == desk_cid
    assert "kb_01" in contexts[desk_cid].member_track_ids

    # Lamp is far away -> belongs to world
    assert membership["lamp_01"] == "world"
    assert "lamp_01" in contexts["world"].member_track_ids

    # Desk itself belongs to world
    assert membership["desk_01"] == "world"
