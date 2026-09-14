from collections import deque
import numpy as np
import pytest

from scene_graph.config import SceneGraphConfig
from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.reference_frame import CameraFrame, RelationReferenceFrame
from scene_graph.relations.context import FrameContext, ObservationGeometry
from scene_graph.relations.directional import DirectionalRelationModule
from scene_graph.relations.evidence import EvidenceResult, ReferenceFrameType
from scene_graph.tracking.track import Track, TrackState


def make_track(object_id: str, centroid: np.ndarray) -> Track:
    return Track(
        object_id=object_id,
        class_name="object",
        state=TrackState.ACTIVE,
        _initial_centroid=np.asarray(centroid, dtype=np.float64),
        last_observed_frame=0,
        first_observed_frame=0,
        observation_count=1,
        missing_count=0,
        detection_confidence=0.9,
        track_observation_ratio=1.0,
        last_timestamp=0.0,
        recent_observations=deque(),
        size_world=np.array([0.2, 0.2, 0.2]),
    )


def make_obs_geom(track_id: str, centroid: np.ndarray) -> ObservationGeometry:
    c = np.asarray(centroid, dtype=np.float64)
    return ObservationGeometry(
        obs_id=f"obs_{track_id}",
        track_id=track_id,
        centroid_world=c,
        bbox_min_world=c - 0.1,
        bbox_max_world=c + 0.1,
        depth_stats={"mean": float(c[2])},
        points_world_sampled=np.array([c]),
        points_world=np.array([c]),
        points_camera=None,
        mask=None,
        valid_point_count=1,
    )


def test_reference_frame_create_factory():
    """Asserts RelationReferenceFrame.create factory constructs valid frames for CAMERA, MAP, GRAVITY."""
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]

    cam_frame = RelationReferenceFrame.create("camera", pose)
    assert cam_frame.frame_type == ReferenceFrameType.CAMERA
    assert np.allclose(cam_frame.origin_world, [1.0, 2.0, 3.0])

    map_frame = RelationReferenceFrame.create("map", pose)
    assert map_frame.frame_type == ReferenceFrameType.WORLD
    assert np.allclose(map_frame.horizontal_axis_world, [1.0, 0.0, 0.0])

    grav_frame = RelationReferenceFrame.create(
        "gravity", pose, gravity_vector=np.array([0.0, 0.0, 1.0]), initial_heading=np.array([1.0, 0.0, 0.0])
    )
    assert grav_frame.frame_type == ReferenceFrameType.WORLD
    assert np.allclose(grav_frame.up_axis_world, [0.0, 0.0, 1.0])


def test_camera_rotation_invariance_under_map_mode():
    """Asserts rotating camera 180 deg changes LEFT_OF in CAMERA mode, but leaves it invariant in MAP mode."""
    config = SceneGraphConfig.from_files("configs/default.yaml")
    module = DirectionalRelationModule(config)

    # Object A is at (-0.5, 2.0, 0.0) -> West / Left in Map frame
    # Object B is at (+0.5, 2.0, 0.0) -> East / Right in Map frame
    track_a = make_track("obj_A", [-0.5, 2.0, 0.0])
    track_b = make_track("obj_B", [0.5, 2.0, 0.0])
    geom_a = make_obs_geom("obj_A", [-0.5, 2.0, 0.0])
    geom_b = make_obs_geom("obj_B", [0.5, 2.0, 0.0])

    # Pose 1: Camera at origin looking along +Y (identity rotation)
    # OpenCV camera frame: X right, Y down, Z forward
    # Looking along +Y means camera Z is +Y, camera Y is -Z (down), camera X is +X (right)
    # Rotation matrix: [ [1, 0, 0], [0, 0, -1], [0, 1, 0] ]
    R1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ])
    T1 = np.eye(4)
    T1[:3, :3] = R1

    # Pose 2: Camera rotated 180 degrees around Z axis (looking along -Y from (0, 4, 0))
    # Camera X is -X (leftwards in map)
    R2 = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ])
    T2 = np.eye(4)
    T2[:3, :3] = R2
    T2[:3, 3] = [0.0, 4.0, 0.0]

    # --- Test under MAP mode ---
    ref_map = RelationReferenceFrame.create("map", T1)

    ctx_map_pose1 = FrameContext(
        frame_index=0,
        timestamp=0.0,
        intrinsics=None,
        world_T_camera=T1,
        reference_frame=ref_map,
        camera_frame=CameraFrame.from_camera_pose(T1),
        depth_image=None,
        observation_geometry={"obj_A": geom_a, "obj_B": geom_b},
    )
    ev_map_1 = module.compute(track_a, track_b, ctx_map_pose1)
    ev_map_1_dict = {e.predicate: e.result for e in ev_map_1}
    assert ev_map_1_dict.get("LEFT_OF") == EvidenceResult.SUPPORTED

    ctx_map_pose2 = FrameContext(
        frame_index=1,
        timestamp=1.0,
        intrinsics=None,
        world_T_camera=T2,
        reference_frame=ref_map,  # Unchanged allocentric map frame!
        camera_frame=CameraFrame.from_camera_pose(T2),
        depth_image=None,
        observation_geometry={"obj_A": geom_a, "obj_B": geom_b},
    )
    ev_map_2 = module.compute(track_a, track_b, ctx_map_pose2)
    ev_map_2_dict = {e.predicate: e.result for e in ev_map_2}
    # Invariant! A is still West/Left of B in MAP frame
    assert ev_map_2_dict.get("LEFT_OF") == EvidenceResult.SUPPORTED

    # --- Test under CAMERA mode ---
    ref_cam_pose1 = RelationReferenceFrame.create("camera", T1)
    ctx_cam_pose1 = FrameContext(
        frame_index=0,
        timestamp=0.0,
        intrinsics=None,
        world_T_camera=T1,
        reference_frame=ref_cam_pose1,
        camera_frame=CameraFrame.from_camera_pose(T1),
        depth_image=None,
        observation_geometry={"obj_A": geom_a, "obj_B": geom_b},
    )
    ev_cam_1 = module.compute(track_a, track_b, ctx_cam_pose1)
    ev_cam_1_dict = {e.predicate: e.result for e in ev_cam_1}
    assert ev_cam_1_dict.get("LEFT_OF") == EvidenceResult.SUPPORTED
    assert ev_cam_1[0].reference_frame == ReferenceFrameType.CAMERA

    # In Pose 2 (viewpoint reversed), A is on the right of B in the camera image!
    ref_cam_pose2 = RelationReferenceFrame.create("camera", T2)
    ctx_cam_pose2 = FrameContext(
        frame_index=1,
        timestamp=1.0,
        intrinsics=None,
        world_T_camera=T2,
        reference_frame=ref_cam_pose2,
        camera_frame=CameraFrame.from_camera_pose(T2),
        depth_image=None,
        observation_geometry={"obj_A": geom_a, "obj_B": geom_b},
    )
    ev_cam_2 = module.compute(track_a, track_b, ctx_cam_pose2)
    ev_cam_2_dict = {e.predicate: e.result for e in ev_cam_2}
    # In camera 2 view, A is NOT LEFT_OF B (it is CONTRADICTED or NOT_APPLICABLE)
    assert ev_cam_2_dict.get("LEFT_OF") == EvidenceResult.CONTRADICTED
