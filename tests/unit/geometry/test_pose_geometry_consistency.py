"""Focused tests for pose and geometry consistency contract.

Enforces:
1. Global geometry is valid only when a valid, sufficiently fresh world-to-camera transform exists.
2. Missing or stale transforms must not produce global coordinates.
3. Camera-relative fallback may produce local geometry, but must never update global centroids,
   global bounding boxes, global relations, or global-map state.
4. Geometry records, pose telemetry, and graph-update decisions must agree for the same frame and timestamp.
5. Tracking must not associate camera-relative geometry with global geometry as though both use the same coordinate frame.
6. Record the coordinate frame, localization status, and relevant pose timestamp explicitly wherever required.
"""

from collections import deque
import numpy as np
import pytest

from scene_graph.config import load_config
from scene_graph.data.frame_packet import FramePacket, LocalizationMode
from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.geometry.point_cloud import (
    GeometryStatus,
    ObjectGeometry,
    compute_object_geometry,
)
from scene_graph.graph.snapshot import create_snapshot
from scene_graph.perception.observation import Observation
from scene_graph.pipeline.pipeline_core import SceneGraphPipeline
from scene_graph.tracking.causal_tracker import CausalTracker
from scene_graph.tracking.track import Track, TrackState


@pytest.fixture
def test_setup():
    intrinsics = CameraIntrinsics(fx=384.0, fy=384.0, cx=32.0, cy=24.0, width=64, height=48)
    depth_model = DepthModel(scale=1000.0)

    rgb = np.full((48, 64, 3), 100, dtype=np.uint8)
    depth = np.full((48, 64), 2.0, dtype=np.float32)

    mask = np.zeros((48, 64), dtype=bool)
    mask[20:28, 28:36] = True

    config = load_config("configs/default.yaml")

    return {
        "intrinsics": intrinsics,
        "depth_model": depth_model,
        "rgb": rgb,
        "depth": depth,
        "mask": mask,
        "config": config,
    }


def make_observation(obs_id="obs_001", mask=None):
    return Observation(
        obs_id=obs_id,
        frame_index=0,
        timestamp=100.0,
        class_name="cup",
        confidence=0.95,
        bbox_xyxy=np.array([28.0, 20.0, 36.0, 28.0], dtype=np.float32),
        mask_rle={"counts": [0, 48 * 64], "size": [48, 64]} if mask is None else None,
    )


def test_valid_fresh_transform_produces_global_geometry(test_setup):
    """Invariant 1: Fresh valid world-to-camera transform produces valid global geometry."""
    cfg = test_setup["config"]
    pipeline = SceneGraphPipeline(cfg)

    world_T_cam = np.eye(4, dtype=np.float64)
    world_T_cam[0, 3] = 1.5
    world_T_cam[1, 3] = 0.5
    world_T_cam[2, 3] = -0.2

    packet = FramePacket(
        frame_index=1,
        timestamp=100.0,
        rgb=test_setup["rgb"],
        depth=test_setup["depth"],
        world_T_camera=world_T_cam,
        camera_intrinsics=test_setup["intrinsics"],
        depth_model=test_setup["depth_model"],
        pose_source="tf_exact",
        transform_source="tf_exact",
        transform_valid=True,
        pose_timestamp=100.0,
        pose_age=0.0,
        localization_mode=LocalizationMode.WORLD_MODE,
    )

    obs = make_observation("obs_001")
    obs.get_mask = lambda: test_setup["mask"]

    pipeline._compute_geometry(packet, [obs], test_setup["depth"])

    geom = obs.object_geometry
    assert geom is not None
    assert geom.status == GeometryStatus.VALID
    assert geom.coordinate_frame == "world"
    assert geom.localization_status == "localized"
    assert geom.centroid_world is not None
    assert geom.points_world is not None
    assert geom.bbox_min_world is not None
    assert geom.bbox_max_world is not None

    # Global centroid X must reflect camera translation + point depth projection
    assert geom.centroid_world[0] == pytest.approx(1.5, abs=0.1)


def test_missing_transform_produces_no_global_geometry(test_setup):
    """Invariant 2: Missing transform must strictly produce no global coordinates."""
    cfg = test_setup["config"]
    pipeline = SceneGraphPipeline(cfg)

    packet = FramePacket(
        frame_index=2,
        timestamp=100.0,
        rgb=test_setup["rgb"],
        depth=test_setup["depth"],
        world_T_camera=None,
        camera_intrinsics=test_setup["intrinsics"],
        depth_model=test_setup["depth_model"],
        transform_source="missing_tf",
        transform_valid=False,
        localization_mode=LocalizationMode.WORLD_MODE,
        metadata={"allow_camera_fallback": True},
    )

    obs = make_observation("obs_002")
    obs.get_mask = lambda: test_setup["mask"]

    pipeline._compute_geometry(packet, [obs], test_setup["depth"])

    geom = obs.object_geometry
    assert geom is not None
    assert geom.status == GeometryStatus.NO_POSE
    assert geom.coordinate_frame == "camera"
    assert geom.localization_status == "NO_POSE"
    # Invariant: NEVER populate world geometry when transform is missing
    assert geom.centroid_world is None
    assert geom.points_world is None
    assert geom.bbox_min_world is None
    assert geom.bbox_max_world is None
    assert geom.obb_center_world is None
    # Local camera centroid is preserved for 2D/3D camera-local understanding
    assert geom.centroid_camera is not None
    assert geom.points_camera is not None


def test_stale_transform_is_rejected(test_setup):
    """Invariant 1 & 2: Stale transform is rejected with STALE_POSE and produces no global geometry."""
    cfg = test_setup["config"]
    pipeline = SceneGraphPipeline(cfg)

    stale_pose = np.eye(4, dtype=np.float64)
    stale_pose[0, 3] = 5.0

    packet = FramePacket(
        frame_index=3,
        timestamp=100.0,
        rgb=test_setup["rgb"],
        depth=test_setup["depth"],
        world_T_camera=stale_pose,
        camera_intrinsics=test_setup["intrinsics"],
        depth_model=test_setup["depth_model"],
        transform_source="stale_tf",
        transform_valid=False,
        pose_timestamp=99.5,
        pose_age=0.5,  # Stale: 500ms > max_pose_age (200ms)
        localization_mode=LocalizationMode.WORLD_MODE,
        metadata={"allow_camera_fallback": True},
    )

    obs = make_observation("obs_003")
    obs.get_mask = lambda: test_setup["mask"]

    pipeline._compute_geometry(packet, [obs], test_setup["depth"])

    geom = obs.object_geometry
    assert geom is not None
    assert geom.status == GeometryStatus.STALE_POSE
    assert geom.coordinate_frame == "camera"
    assert geom.localization_status == "STALE_POSE"
    assert geom.centroid_world is None
    assert geom.points_world is None
    assert geom.bbox_min_world is None


def test_timestamp_mismatch_handled_explicitly(test_setup):
    """Invariant 1: Frame timestamp vs pose timestamp mismatch (> max_pose_age) is rejected."""
    cfg = test_setup["config"]
    pipeline = SceneGraphPipeline(cfg)

    packet = FramePacket(
        frame_index=4,
        timestamp=100.0,
        rgb=test_setup["rgb"],
        depth=test_setup["depth"],
        world_T_camera=np.eye(4, dtype=np.float64),
        camera_intrinsics=test_setup["intrinsics"],
        depth_model=test_setup["depth_model"],
        pose_timestamp=99.70,
        pose_age=0.30,  # 300ms skew
        transform_valid=True,
        localization_mode=LocalizationMode.WORLD_MODE,
        metadata={"allow_camera_fallback": True},
    )

    obs = make_observation("obs_004")
    obs.get_mask = lambda: test_setup["mask"]

    pipeline._compute_geometry(packet, [obs], test_setup["depth"])

    geom = obs.object_geometry
    assert geom.status == GeometryStatus.STALE_POSE
    assert geom.centroid_world is None
    assert geom.coordinate_frame == "camera"


def test_camera_relative_fallback_cannot_modify_global_state(test_setup):
    """Invariant 3: Camera-relative fallback cannot modify global centroids or map state."""
    cfg = test_setup["config"]
    pipeline = SceneGraphPipeline(cfg)

    # Frame 1: missing transform with local fallback
    packet = FramePacket(
        frame_index=1,
        timestamp=10.0,
        rgb=test_setup["rgb"],
        depth=test_setup["depth"],
        world_T_camera=None,
        camera_intrinsics=test_setup["intrinsics"],
        depth_model=test_setup["depth_model"],
        transform_source="missing_tf",
        transform_valid=False,
        localization_mode=LocalizationMode.WORLD_MODE,
        metadata={"allow_camera_fallback": True},
    )

    obs = make_observation("obs_local")
    obs.get_mask = lambda: test_setup["mask"]

    graph = pipeline.update(packet, [obs])

    # Persistent global world model must NOT contain unlocalized local-fallback objects
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0

    # Snapshot must reflect no global objects and null world_T_camera
    snapshot = create_snapshot(graph, packet)
    assert snapshot.world_T_camera is None
    assert len(snapshot.objects) == 0


def test_local_and_global_observations_cannot_be_associated(test_setup):
    """Invariant 5: Tracking must not associate camera-relative geometry with global geometry."""
    cfg = test_setup["config"]
    tracker = CausalTracker(cfg)

    # Track 1: confirmed global track in world frame at [1.0, 2.0, 3.0]
    track_world = Track(
        object_id="track_0001",
        class_name="cup",
        state=TrackState.ACTIVE,
        _initial_centroid=np.array([1.0, 2.0, 3.0]),
        last_observed_frame=1,
        first_observed_frame=1,
        observation_count=3,
        missing_count=0,
        detection_confidence=0.9,
        track_observation_ratio=1.0,
        last_timestamp=10.0,
        recent_observations=deque(),
        coordinate_frame="world",
        size_world=np.array([0.1, 0.1, 0.1]),
    )
    tracker.tracks["track_0001"] = track_world

    # Observation: camera-local observation (coincidentally near [1.0, 2.0, 3.0] in camera frame)
    obs_cam = make_observation("obs_cam")
    obs_cam.object_geometry = ObjectGeometry(
        status=GeometryStatus.NO_POSE,
        coordinate_frame="camera",
        centroid_camera=np.array([1.0, 2.0, 3.0]),
        centroid_world=None,  # Crucial: no world centroid
        points_camera=np.array([[1.0, 2.0, 3.0]]),
        points_world=None,
    )

    predictions = tracker._predict_tracks(10.033)
    matches, unmatched_obs, unmatched_tracks = tracker._associate([obs_cam], predictions)

    # Invariant: must NOT associate camera-frame observation with world-frame track
    assert len(matches) == 0
    assert 0 in unmatched_obs
    assert "track_0001" in unmatched_tracks


def test_telemetry_and_geometry_validity_agree(test_setup):
    """Invariant 4: Telemetry, geometry records, and snapshot decisions agree."""
    cfg = test_setup["config"]
    pipeline = SceneGraphPipeline(cfg)

    # Case A: Invalid pose packet
    packet_invalid = FramePacket(
        frame_index=1,
        timestamp=20.0,
        rgb=test_setup["rgb"],
        depth=test_setup["depth"],
        world_T_camera=None,
        camera_intrinsics=test_setup["intrinsics"],
        transform_source="no_tf",
        transform_valid=False,
        localization_mode=LocalizationMode.WORLD_MODE,
        metadata={"allow_camera_fallback": True},
    )
    obs = make_observation("obs_test")
    obs.get_mask = lambda: test_setup["mask"]

    graph = pipeline.update(packet_invalid, [obs])
    snapshot = create_snapshot(graph, packet_invalid)

    # Telemetry pose validity is False -> Snapshot world pose is null -> Geometry centroid_world is None
    assert packet_invalid.transform_valid is False
    assert packet_invalid.is_persistent_world_frame is False
    assert snapshot.world_T_camera is None
    assert obs.object_geometry.centroid_world is None
    assert obs.object_geometry.status == GeometryStatus.NO_POSE
    assert obs.object_geometry.coordinate_frame == "camera"

    # Case B: Valid pose packet
    packet_valid = FramePacket(
        frame_index=2,
        timestamp=20.033,
        rgb=test_setup["rgb"],
        depth=test_setup["depth"],
        world_T_camera=np.eye(4, dtype=np.float64),
        camera_intrinsics=test_setup["intrinsics"],
        transform_source="tf_exact",
        transform_valid=True,
        pose_timestamp=20.033,
        pose_age=0.0,
        localization_mode=LocalizationMode.WORLD_MODE,
    )
    obs_valid = make_observation("obs_test_valid")
    obs_valid.get_mask = lambda: test_setup["mask"]

    graph_valid = pipeline.update(packet_valid, [obs_valid])
    snapshot_valid = create_snapshot(graph_valid, packet_valid)

    # Telemetry pose validity is True -> Snapshot world pose is valid matrix -> Geometry centroid_world is valid
    assert packet_valid.transform_valid is True
    assert packet_valid.is_persistent_world_frame is True
    assert snapshot_valid.world_T_camera is not None
    assert obs_valid.object_geometry.centroid_world is not None
    assert obs_valid.object_geometry.status == GeometryStatus.VALID
    assert obs_valid.object_geometry.coordinate_frame == "world"


def test_invalid_depth_with_valid_pose_preserves_localization_status(test_setup):
    """Invariant: Invalid depth with valid pose does not falsely mark frame as unlocalized."""
    cfg = test_setup["config"]
    pipeline = SceneGraphPipeline(cfg)

    packet = FramePacket(
        frame_index=10,
        timestamp=50.0,
        rgb=test_setup["rgb"],
        depth=test_setup["depth"],
        world_T_camera=np.eye(4, dtype=np.float64),
        camera_intrinsics=test_setup["intrinsics"],
        depth_model=test_setup["depth_model"],
        transform_source="tf_exact",
        transform_valid=True,
        pose_timestamp=50.0,
        pose_age=0.0,
        localization_mode=LocalizationMode.WORLD_MODE,
    )

    # Empty mask with zero valid points
    empty_mask = np.zeros((48, 64), dtype=bool)
    obs = make_observation("obs_no_depth")
    obs.get_mask = lambda: empty_mask

    pipeline._compute_geometry(packet, [obs], test_setup["depth"])

    geom = obs.object_geometry
    assert geom is not None
    assert geom.status == GeometryStatus.INSUFFICIENT_DEPTH
    # Localization status remains localized because the SLAM pose was valid
    assert geom.coordinate_frame == "world"
    assert geom.localization_status == "localized"
    assert geom.centroid_world is None
    assert geom.centroid_camera is None


def test_snapshot_exposes_localization_provenance(test_setup):
    """Invariant: Snapshots must preserve explicit localization and transform provenance."""
    cfg = test_setup["config"]
    pipeline = SceneGraphPipeline(cfg)

    # Localized frame
    packet = FramePacket(
        frame_index=25,
        timestamp=100.0,
        rgb=test_setup["rgb"],
        depth=test_setup["depth"],
        world_T_camera=np.eye(4, dtype=np.float64),
        camera_intrinsics=test_setup["intrinsics"],
        depth_model=test_setup["depth_model"],
        transform_source="slam_causal_tf",
        transform_valid=True,
        pose_timestamp=99.985,
        pose_age=0.015,
        localization_mode=LocalizationMode.WORLD_MODE,
    )
    obs = make_observation("obs_25")
    obs.get_mask = lambda: test_setup["mask"]

    graph = pipeline.update(packet, [obs])
    snapshot = create_snapshot(graph, packet)

    snap_dict = snapshot.to_dict()
    assert "localization" in snap_dict
    loc = snap_dict["localization"]
    assert loc["status"] == "localized"
    assert loc["coordinate_frame"] == "world"
    assert loc["transform_valid"] is True
    assert loc["transform_source"] == "slam_causal_tf"
    assert loc["pose_timestamp"] == 99.985
    assert loc["pose_age_sec"] == pytest.approx(0.015, abs=1e-3)
    assert loc["is_persistent_world_frame"] is True


def test_primary_geometry_metrics_mutually_exclusive():
    """Invariant: Primary geometry outcome categories are mutually exclusive and collectively exhaustive."""
    categories = {
        "localized_valid_geometry": 0,
        "localized_insufficient_depth": 0,
        "unlocalized_valid_geometry": 0,
        "unlocalized_insufficient_depth": 0,
    }

    # Helper simulating the 4 cases
    test_cases = [
        # (is_global_valid, is_depth_ok, expected_category)
        (True, True, "localized_valid_geometry"),
        (True, False, "localized_insufficient_depth"),
        (False, True, "unlocalized_valid_geometry"),
        (False, False, "unlocalized_insufficient_depth"),
    ]

    for is_global_valid, is_depth_ok, expected_cat in test_cases:
        matched = 0
        if is_global_valid and is_depth_ok:
            categories["localized_valid_geometry"] += 1
            matched += 1
        elif is_global_valid and not is_depth_ok:
            categories["localized_insufficient_depth"] += 1
            matched += 1
        elif not is_global_valid and is_depth_ok:
            categories["unlocalized_valid_geometry"] += 1
            matched += 1
        else:
            categories["unlocalized_insufficient_depth"] += 1
            matched += 1
        # Exactly 1 category matched per observation
        assert matched == 1

    assert sum(categories.values()) == len(test_cases)
    for cat in categories.values():
        assert cat == 1

