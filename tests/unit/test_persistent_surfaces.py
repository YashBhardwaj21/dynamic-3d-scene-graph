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
