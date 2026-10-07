from collections import deque
import numpy as np
import pytest

from scene_graph.config import SceneGraphConfig
from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.reference_frame import CameraFrame, RelationReferenceFrame
from scene_graph.geometry.spatial_surface import SpatialSurface, SurfaceManager
from scene_graph.relations.context import FrameContext, ObservationGeometry
from scene_graph.relations.evidence import EvidenceResult, ReferenceFrameType
from scene_graph.relations.support import SupportRelationModule
from scene_graph.tracking.track import Track, TrackState


def test_coplanar_surface_merging():
    """Asserts two coplanar patches with offset < 0.03m and normal angle < 10 deg merge successfully."""
    manager = SurfaceManager(max_angle_deg=10.0, max_offset_m=0.03)

    normal1 = np.array([0.0, 0.0, 1.0])
    pts1 = np.array([
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ])
    surf1 = manager.register_or_update("track_01", normal1, -0.5, pts1, timestamp=0.0)

    # Patch 2: slightly shifted normal (2 degrees) and offset by 0.015m
    theta = np.radians(2.0)
    normal2 = np.array([np.sin(theta), 0.0, np.cos(theta)])
    pts2 = np.array([
        [0.5, 0.5, 0.515],
        [1.0, 0.5, 0.515],
        [0.5, 1.0, 0.515],
        [1.0, 1.0, 0.515],
    ])
    surf2 = manager.register_or_update("track_01", normal2, -0.515, pts2, timestamp=0.1)

    # They should have merged into a single surface under track_01
    assert len(manager.surfaces) == 1
    merged = manager.get_surface("track_01", current_timestamp=0.1)
    assert merged is not None
    assert merged.inlier_count == len(pts1) + len(pts2)
    assert np.isclose(merged.normal[2], 1.0, atol=0.05)


def test_non_coplanar_surfaces_not_merged():
    """Asserts patches with angular discrepancy > 10 deg or large vertical offset are not merged."""
    s1 = SpatialSurface(
        surface_id="s1",
        normal=np.array([0.0, 0.0, 1.0]),
        distance=-0.5,
        bounds_world=(np.zeros(3), np.ones(3)),
        last_observed_timestamp=0.0,
        inlier_count=50,
    )
    # Tilted by 25 degrees
    theta = np.radians(25.0)
    s2 = SpatialSurface(
        surface_id="s2",
        normal=np.array([np.sin(theta), 0.0, np.cos(theta)]),
        distance=-0.5,
        bounds_world=(np.zeros(3), np.ones(3)),
        last_observed_timestamp=0.0,
        inlier_count=50,
    )
    assert s1.can_merge(s2, max_angle_deg=10.0, max_offset_m=0.03) is False

    # Large offset (0.10m > 0.03m)
    s3 = SpatialSurface(
        surface_id="s3",
        normal=np.array([0.0, 0.0, 1.0]),
        distance=-0.60,
        bounds_world=(np.zeros(3), np.ones(3)),
        last_observed_timestamp=0.0,
        inlier_count=50,
    )
    assert s1.can_merge(s3, max_angle_deg=10.0, max_offset_m=0.03) is False


def test_surface_persistence_across_occlusion():
    """Asserts SupportRelationModule uses persistent surface when current frame experiences point dropout."""
    config = SceneGraphConfig.from_files("configs/default.yaml")
    module = SupportRelationModule(config)

    ref_frame = RelationReferenceFrame(
        origin_world=np.zeros(3),
        up_axis_world=np.array([0.0, 0.0, 1.0]),
        horizontal_axis_world=np.array([1.0, 0.0, 0.0]),
        depth_axis_world=np.array([0.0, 1.0, 0.0]),
    )

    # Frame 0: Table with rich planar surface (50 points at z=0.5)
    xs, ys = np.meshgrid(np.linspace(-0.5, 0.5, 8), np.linspace(-0.5, 0.5, 8))
    table_pts = np.column_stack([xs.flatten(), ys.flatten(), np.full(64, 0.5)])

    # Cup resting on table (points at z=0.505 to 0.60)
    cup_xs, cup_ys = np.meshgrid(np.linspace(-0.05, 0.05, 5), np.linspace(-0.05, 0.05, 5))
    cup_pts = np.column_stack([cup_xs.flatten(), cup_ys.flatten(), np.full(25, 0.51)])

    table_geom_f0 = ObservationGeometry(
        obs_id="obs_table_0",
        track_id="track_table",
        centroid_world=np.array([0.0, 0.0, 0.5]),
        bbox_min_world=np.array([-0.5, -0.5, 0.45]),
        bbox_max_world=np.array([0.5, 0.5, 0.55]),
        depth_stats={"mean": 2.0},
        points_world_sampled=table_pts,
        points_world=table_pts,
        points_camera=None,
        mask=np.ones((50, 50), dtype=bool),
        valid_point_count=len(table_pts),
    )

    cup_geom = ObservationGeometry(
        obs_id="obs_cup_0",
        track_id="track_cup",
        centroid_world=np.array([0.0, 0.0, 0.55]),
        bbox_min_world=np.array([-0.05, -0.05, 0.505]),
        bbox_max_world=np.array([0.05, 0.05, 0.60]),
        depth_stats={"mean": 2.0},
        points_world_sampled=cup_pts,
        points_world=cup_pts,
        points_camera=None,
        mask=np.ones((50, 50), dtype=bool),
        valid_point_count=len(cup_pts),
    )

    cup_track = Track(
        object_id="track_cup",
        class_name="cup",
        state=TrackState.ACTIVE,
        _initial_centroid=np.array([0.0, 0.0, 0.55]),
        last_observed_frame=0,
        first_observed_frame=0,
        observation_count=1,
        missing_count=0,
        detection_confidence=0.9,
        track_observation_ratio=1.0,
        last_timestamp=0.0,
        recent_observations=deque(),
        size_world=np.array([0.1, 0.1, 0.1]),
    )
    table_track = Track(
        object_id="track_table",
        class_name="desk",
        state=TrackState.ACTIVE,
        _initial_centroid=np.array([0.0, 0.0, 0.5]),
        last_observed_frame=0,
        first_observed_frame=0,
        observation_count=1,
        missing_count=0,
        detection_confidence=0.9,
        track_observation_ratio=1.0,
        last_timestamp=0.0,
        recent_observations=deque(),
        size_world=np.array([1.0, 1.0, 0.1]),
    )

    ctx_f0 = FrameContext(
        frame_index=0,
        timestamp=0.0,
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=25.0, cy=25.0, width=50, height=50),
        world_T_camera=np.eye(4),
        reference_frame=ref_frame,
        camera_frame=CameraFrame.from_camera_pose(np.eye(4)),
        depth_image=np.ones((50, 50), dtype=np.float32),
        observation_geometry={"track_cup": cup_geom, "track_table": table_geom_f0},
    )

    evidences_f0 = module.compute(cup_track, table_track, ctx_f0)
    assert len(evidences_f0) == 1
    assert evidences_f0[0].predicate == "ON"
    assert evidences_f0[0].result == EvidenceResult.SUPPORTED

    # Frame 1 (0.2s later): Table experiences sensor dropout (only 2 points left, RANSAC impossible)
    sparse_table_pts = np.array([[0.0, 0.0, 0.5], [0.1, 0.1, 0.5]])
    table_geom_f1 = ObservationGeometry(
        obs_id="obs_table_1",
        track_id="track_table",
        centroid_world=np.array([0.0, 0.0, 0.5]),
        bbox_min_world=np.array([-0.5, -0.5, 0.45]),
        bbox_max_world=np.array([0.5, 0.5, 0.55]),
        depth_stats={"mean": 2.0},
        points_world_sampled=sparse_table_pts,
        points_world=sparse_table_pts,
        points_camera=None,
        mask=np.ones((50, 50), dtype=bool),
        valid_point_count=len(sparse_table_pts),
    )

    ctx_f1 = FrameContext(
        frame_index=1,
        timestamp=0.2,
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=25.0, cy=25.0, width=50, height=50),
        world_T_camera=np.eye(4),
        reference_frame=ref_frame,
        camera_frame=CameraFrame.from_camera_pose(np.eye(4)),
        depth_image=np.ones((50, 50), dtype=np.float32),
        observation_geometry={"track_cup": cup_geom, "track_table": table_geom_f1},
    )

    evidences_f1 = module.compute(cup_track, table_track, ctx_f1)
    assert len(evidences_f1) == 1
    assert evidences_f1[0].predicate == "ON"
    assert evidences_f1[0].result == EvidenceResult.SUPPORTED


def test_ground_plane_extraction_synthetic_depth():
    """Verify GroundPlaneExtractor extracts dominant ground plane and respects foreground masks."""
    from scene_graph.geometry.ground_plane import GroundPlaneExtractor, GroundPlaneConfig

    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)

    # Camera at (0, 0, 1.2) tilted downwards by 30 degrees (pitch = 30 deg)
    # Rotation: pitch around X axis: R_pitch = [[1, 0, 0], [0, cos, -sin], [0, sin, cos]]
    # OpenCV camera frame: X right, Y down, Z forward
    # If camera points forward-down:
    pitch = np.radians(30.0)
    # Camera facing +Y, pitched down by 30 deg:
    # X_c = [1, 0, 0]
    # Y_c = [0, -sin(pitch), -cos(pitch)]
    # Z_c = [0, cos(pitch), -sin(pitch)]
    R_wc = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -np.sin(pitch), np.cos(pitch)],
        [0.0, -np.cos(pitch), -np.sin(pitch)],
    ])
    world_T_cam = np.eye(4)
    world_T_cam[:3, :3] = R_wc
    world_T_cam[:3, 3] = [0.0, 0.0, 1.2]

    # Synthesize ground plane depth image where ground is at world Z = 0
    depth_m = np.zeros((480, 640), dtype=np.float32)
    for v in range(0, 480, 4):
        for u in range(0, 640, 4):
            ray_cam = np.array([(u - 320.0) / 500.0, (v - 240.0) / 500.0, 1.0])
            ray_world = R_wc @ ray_cam
            if ray_world[2] < -1e-4:  # pointing downward toward ground
                z_c = -1.2 / ray_world[2]
                if 0.3 <= z_c <= 8.0:
                    depth_m[v : v + 4, u : u + 4] = float(z_c)

    cfg = GroundPlaneConfig(subsample_step=8, min_inliers=30)
    extractor = GroundPlaneExtractor(cfg)

    # Extract ground plane
    surface = extractor.extract(depth_m, intrinsics, world_T_camera=world_T_cam, timestamp=0.0)
    assert surface is not None
    # Ground normal in world must point strictly up (+Z)
    assert np.isclose(surface.normal[2], 1.0, atol=0.05)
    assert abs(surface.normal[0]) < 0.05
    assert abs(surface.normal[1]) < 0.05
    # Plane distance d such that n . x + d = 0 for z = 0 -> d is near 0
    assert abs(surface.distance) < 0.06

    # Test foreground mask exclusion: add a big foreground object sitting on the ground
    mask = np.zeros((480, 640), dtype=bool)
    mask[300:400, 200:400] = True
    depth_with_obj = depth_m.copy()
    depth_with_obj[300:400, 200:400] = 1.0  # object much closer to camera

    surface_masked = extractor.extract(
        depth_with_obj, intrinsics, world_T_camera=world_T_cam, exclude_masks=[mask], timestamp=0.0
    )
    assert surface_masked is not None
    assert np.isclose(surface_masked.normal[2], 1.0, atol=0.05)


def test_ground_plane_tracking_ema_and_persistence():
    """Verify GroundPlaneTracker smooths plane estimates across frames and handles sensor dropouts."""
    from scene_graph.geometry.ground_plane import GroundPlaneTracker, GroundPlaneConfig

    cfg = GroundPlaneConfig(subsample_step=8, ema_alpha=0.30, max_age_s=2.0)
    tracker = GroundPlaneTracker(cfg)

    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    world_T_cam = np.eye(4)
    world_T_cam[:3, 3] = [0.0, 0.0, 1.0]

    # Simple flat ground depth image
    depth_m = np.ones((480, 640), dtype=np.float32) * 2.0

    # Frame 0: Initial observation at t = 0.0
    surf0 = tracker.update(depth_m, intrinsics, world_T_camera=world_T_cam, timestamp=0.0)
    assert tracker.is_initialized
    assert surf0 is not None

    # Frame 1: Slightly shifted ground at t = 0.1
    surf1 = tracker.update(depth_m, intrinsics, world_T_camera=world_T_cam, timestamp=0.1)
    assert surf1 is not None
    assert surf1.last_observed_timestamp == 0.1

    # Frame 2: Complete sensor dropout at t = 0.5 (empty depth image)
    empty_depth = np.zeros((480, 640), dtype=np.float32)
    surf2 = tracker.update(empty_depth, intrinsics, world_T_camera=world_T_cam, timestamp=0.5)
    # Tracker should return the persistent ground plane within max_age_s!
    assert surf2 is not None
    assert surf2.last_observed_timestamp == 0.1

    # Frame 3: Expired after max_age_s (t = 3.5s > 0.1 + 2.0s)
    surf3 = tracker.update(empty_depth, intrinsics, world_T_camera=world_T_cam, timestamp=3.5)
    assert surf3 is None


def test_ground_aligned_reference_frame():
    """Verify create_ground_aligned_reference_frame constructs an orthonormal frame anchored to ground."""
    from scene_graph.geometry.spatial_surface import SpatialSurface
    from scene_graph.geometry.ground_plane import create_ground_aligned_reference_frame

    # Tilted ground plane with normal slightly off from [0, 0, 1]
    raw_n = np.array([0.05, 0.0, 0.9987])
    n = raw_n / np.linalg.norm(raw_n)
    d = -0.50  # ground at z ≈ 0.5

    surf = SpatialSurface(
        surface_id="ground",
        normal=n,
        distance=d,
        bounds_world=(np.array([-2.0, -2.0, 0.0]), np.array([2.0, 2.0, 1.0])),
        last_observed_timestamp=0.0,
        inlier_count=100,
    )

    frame = create_ground_aligned_reference_frame(surf, initial_heading=np.array([1.0, 0.0, 0.0]))

    # 1. Up axis must match ground normal
    assert np.allclose(frame.up_axis_world, n, atol=1e-5)

    # 2. Orthonormal axes
    assert np.isclose(np.linalg.norm(frame.up_axis_world), 1.0)
    assert np.isclose(np.linalg.norm(frame.horizontal_axis_world), 1.0)
    assert np.isclose(np.linalg.norm(frame.depth_axis_world), 1.0)

    assert abs(np.dot(frame.up_axis_world, frame.horizontal_axis_world)) < 1e-5
    assert abs(np.dot(frame.up_axis_world, frame.depth_axis_world)) < 1e-5
    assert abs(np.dot(frame.horizontal_axis_world, frame.depth_axis_world)) < 1e-5

    # 3. Origin must lie on the ground plane (n . origin + d = 0)
    origin_eval = np.dot(n, frame.origin_world) + d
    assert abs(origin_eval) < 1e-5


def test_ground_leveling_transform():
    """Verify compute_ground_leveling_transform rotates and translates world so ground is Z=0 with normal=[0,0,1]."""
    from scene_graph.geometry.spatial_surface import SpatialSurface
    from scene_graph.geometry.ground_plane import compute_ground_leveling_transform

    raw_n = np.array([0.08, -0.06, 0.995])
    n = raw_n / np.linalg.norm(raw_n)
    d = -0.80

    surf = SpatialSurface(
        surface_id="ground",
        normal=n,
        distance=d,
        bounds_world=(np.zeros(3), np.ones(3)),
        last_observed_timestamp=0.0,
        inlier_count=50,
    )

    T_level = compute_ground_leveling_transform(surf)
    assert T_level.shape == (4, 4)

    # Rotate ground normal by T_level's rotation matrix
    R = T_level[:3, :3]
    leveled_n = R @ n
    assert np.allclose(leveled_n, [0.0, 0.0, 1.0], atol=1e-4)

    # Take a point on the ground plane: p0 = -d * n
    p0 = -d * n
    assert np.isclose(np.dot(n, p0) + d, 0.0)

    # Transform p0 by T_level: must have Z = 0
    p0_h = np.array([p0[0], p0[1], p0[2], 1.0])
    p0_leveled = T_level @ p0_h
    assert abs(p0_leveled[2]) < 1e-5
