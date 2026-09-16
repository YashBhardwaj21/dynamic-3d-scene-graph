import numpy as np
import pytest
from collections import deque

from scene_graph.config import SceneGraphConfig
from scene_graph.tracking.track import Track, TrackState
from scene_graph.relations.context import FrameContext, ObservationGeometry
from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.reference_frame import RelationReferenceFrame, CameraFrame
from scene_graph.relations.hierarchy import SpatialContext, SpatialContextManager
from scene_graph.relations.candidate_generator import RelationCandidateGenerator


def make_track(obj_id: str, label: str, centroid: np.ndarray) -> Track:
    return Track(
        object_id=obj_id,
        class_name=label,
        state=TrackState.ACTIVE,
        _initial_centroid=centroid,
        last_observed_frame=1,
        first_observed_frame=1,
        observation_count=10,
        missing_count=0,
        detection_confidence=0.9,
        track_observation_ratio=1.0,
        last_timestamp=1.0,
        recent_observations=deque(),
    )


def make_geom(obj_id: str, bbox_min: np.ndarray, bbox_max: np.ndarray) -> ObservationGeometry:
    centroid = (bbox_min + bbox_max) / 2.0
    # Add simple points
    xs = np.linspace(bbox_min[0], bbox_max[0], 5)
    ys = np.linspace(bbox_min[1], bbox_max[1], 5)
    zs = np.linspace(bbox_min[2], bbox_max[2], 5)
    gx, gy, gz = np.meshgrid(xs, ys, zs)
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    return ObservationGeometry(
        obs_id=f"obs_{obj_id}",
        track_id=obj_id,
        centroid_world=centroid,
        bbox_min_world=bbox_min,
        bbox_max_world=bbox_max,
        depth_stats=None,
        points_world_sampled=points,
        points_world=points,
        points_camera=points,  # mock
        mask=None,
        valid_point_count=len(points),
    )


def test_candidate_generator_numerical_reduction():
    """
    Demonstrates numerical candidate reduction:
    10 objects (1 desk + 9 items on desk)
    90 potential pairs -> <= 30 total typed candidates.
    No anchor-child directional pairs!
    """
    # 1. Desk (anchor surface)
    desk = make_track("desk_01", "desk", np.array([0.0, 1.0, 0.0]))
    desk_geom = make_geom("desk_01", np.array([-1.0, 0.4, -0.2]), np.array([1.0, 1.6, 0.0]))

    # 2. 9 items placed on the desk
    # Row 1 (back): monitor, speakers
    monitor = make_track("monitor_01", "monitor", np.array([0.0, 1.3, 0.25]))
    monitor_geom = make_geom("monitor_01", np.array([-0.3, 1.2, 0.0]), np.array([0.3, 1.4, 0.50]))

    speaker_l = make_track("speaker_01", "speaker", np.array([-0.5, 1.3, 0.15]))
    speaker_l_geom = make_geom("speaker_01", np.array([-0.6, 1.25, 0.0]), np.array([-0.4, 1.35, 0.30]))

    speaker_r = make_track("speaker_02", "speaker", np.array([0.5, 1.3, 0.15]))
    speaker_r_geom = make_geom("speaker_02", np.array([0.4, 1.25, 0.0]), np.array([0.6, 1.35, 0.30]))

    # Row 2 (middle): keyboard, mouse
    keyboard = make_track("keyboard_01", "keyboard", np.array([-0.05, 0.8, 0.04]))
    keyboard_geom = make_geom("keyboard_01", np.array([-0.25, 0.7, 0.0]), np.array([0.15, 0.9, 0.08]))

    mouse = make_track("mouse_01", "mouse", np.array([0.30, 0.8, 0.03]))
    mouse_geom = make_geom("mouse_01", np.array([0.25, 0.75, 0.0]), np.array([0.35, 0.85, 0.06]))

    # Row 3 (front/sides): book, cup, pen, phone
    book = make_track("book_01", "book", np.array([-0.60, 0.8, 0.02]))
    book_geom = make_geom("book_01", np.array([-0.75, 0.7, 0.0]), np.array([-0.45, 0.9, 0.04]))

    cup = make_track("cup_01", "cup", np.array([0.50, 0.9, 0.08]))
    cup_geom = make_geom("cup_01", np.array([0.45, 0.85, 0.0]), np.array([0.55, 0.95, 0.16]))

    pen = make_track("pen_01", "pen", np.array([-0.10, 0.6, 0.01]))
    pen_geom = make_geom("pen_01", np.array([-0.18, 0.58, 0.0]), np.array([-0.02, 0.62, 0.02]))

    phone = make_track("phone_01", "cellphone", np.array([0.20, 0.6, 0.01]))
    phone_geom = make_geom("phone_01", np.array([0.15, 0.55, 0.0]), np.array([0.25, 0.65, 0.02]))

    tracks = [desk, monitor, speaker_l, speaker_r, keyboard, mouse, book, cup, pen, phone]
    geoms = {
        "desk_01": desk_geom,
        "monitor_01": monitor_geom,
        "speaker_01": speaker_l_geom,
        "speaker_02": speaker_r_geom,
        "keyboard_01": keyboard_geom,
        "mouse_01": mouse_geom,
        "book_01": book_geom,
        "cup_01": cup_geom,
        "pen_01": pen_geom,
        "phone_01": phone_geom,
    }

    assert len(tracks) == 10

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

    # 1. Discover contexts
    ctx_mgr = SpatialContextManager()
    contexts = ctx_mgr.discover_contexts(tracks, frame_ctx)
    membership = ctx_mgr.assign_context_membership(tracks, contexts, frame_ctx)

    assert "context_desk_01" in contexts

    # 2. Candidate generation
    cand_gen = RelationCandidateGenerator()
    candidates, summary = cand_gen.generate_all_candidates(tracks, frame_ctx, contexts, membership)

    print("\nCandidate Reduction Debug Output:")
    print(str(summary))

    # Verification of key architectural criteria:
    # 1. Total objects = 10, Potential all-pairs = 90
    assert summary.total_objects == 10
    assert summary.potential_pairs == 90

    # 2. Structural candidates: Only objects tested against desk anchor
    assert summary.structural_pairs <= 9
    for sub, obj in candidates["structural"]:
        assert obj.object_id == "desk_01", f"Target of structural must be anchor, got {obj.object_id}"
        assert sub.object_id != "desk_01"

    # 3. No anchor-child in directional pairs (Desk is NEVER LEFT_OF / RIGHT_OF Book/Keyboard)
    for sub, obj in candidates["directional"]:
        assert sub.object_id != "desk_01", "Anchor should never be subject of directional relations"
        assert obj.object_id != "desk_01", "Anchor should never be object of directional relations"

    # 4. No anchor-child in proximity pairs
    for sub, obj in candidates["proximity"]:
        assert sub.object_id != "desk_01", "Anchor should not have proximity relations to children"
        assert obj.object_id != "desk_01", "Anchor should not have proximity relations to children"

    # 5. Numerical reduction: Total candidates <= 30 (from 90 potential pairs, >65% reduction)
    assert summary.total_candidates <= 30, f"Expected <= 30 candidates, got {summary.total_candidates}"
