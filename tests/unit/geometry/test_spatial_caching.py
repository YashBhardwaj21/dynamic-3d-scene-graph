from collections import deque
from unittest.mock import patch
import numpy as np
import pytest

from scene_graph.config import SceneGraphConfig
from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.reference_frame import CameraFrame, RelationReferenceFrame
from scene_graph.relations.context import FrameContext, ObservationGeometry
from scene_graph.relations.distance import DistanceRelationModule
from scene_graph.relations.registry import RelationRegistry
from scene_graph.relations.support import SupportRelationModule
from scene_graph.tracking.track import Track, TrackState


def make_track(object_id: str, centroid: np.ndarray, class_name: str = "object") -> Track:
    return Track(
        object_id=object_id,
        class_name=class_name,
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


def make_obs_geom(track_id: str, pts: np.ndarray) -> ObservationGeometry:
    pts_arr = np.asarray(pts, dtype=np.float64)
    c = np.mean(pts_arr, axis=0)
    return ObservationGeometry(
        obs_id=f"obs_{track_id}",
        track_id=track_id,
        centroid_world=c,
        bbox_min_world=np.min(pts_arr, axis=0),
        bbox_max_world=np.max(pts_arr, axis=0),
        depth_stats={"mean": float(c[2])},
        points_world_sampled=pts_arr,
        points_world=pts_arr,
        points_camera=None,
        mask=np.ones((20, 20), dtype=bool),
        valid_point_count=len(pts_arr),
    )


def test_ckdtree_cached_per_object_per_frame():
    """Asserts cKDTree is instantiated at most N times (once per target object) across N*(N-1) pairs."""
    config = SceneGraphConfig.from_files("configs/default.yaml")
    module = DistanceRelationModule(config)

    # 5 objects -> 20 ordered pairs
    N = 5
    tracks = []
    geometries = {}
    for i in range(N):
        tid = f"obj_{i}"
        pts = np.random.default_rng(i).uniform(i, i + 0.5, size=(30, 3))
        tracks.append(make_track(tid, [i, i, i]))
        geometries[tid] = make_obs_geom(tid, pts)

    ref_frame = RelationReferenceFrame.create("map", np.eye(4))
    ctx = FrameContext(
        frame_index=0,
        timestamp=0.0,
        intrinsics=None,
        world_T_camera=np.eye(4),
        reference_frame=ref_frame,
        camera_frame=CameraFrame.from_camera_pose(np.eye(4)),
        depth_image=None,
        observation_geometry=geometries,
    )

    with patch("scene_graph.relations.distance.cKDTree", wraps=module._tree_cache.get) as mock_tree:
        # Evaluate all 20 pairs
        pair_count = 0
        from scipy.spatial import cKDTree as RealKDTree
        with patch("scene_graph.relations.distance.cKDTree", side_effect=RealKDTree) as real_mock:
            for s in tracks:
                for o in tracks:
                    if s.object_id != o.object_id:
                        module.compute(s, o, ctx)
                        pair_count += 1

            assert pair_count == N * (N - 1)
            # Tree must have been built at most N times (one per object_), NOT 20 times!
            assert real_mock.call_count <= N


def test_plane_ransac_cached_for_multiple_cups_on_table():
    """Asserts fit_plane_ransac runs at most once when 3 cups are evaluated against 1 table in same frame."""
    config = SceneGraphConfig.from_files("configs/default.yaml")
    module = SupportRelationModule(config)

    # Table with 64 planar points at z=0.5
    xs, ys = np.meshgrid(np.linspace(-0.5, 0.5, 8), np.linspace(-0.5, 0.5, 8))
    table_pts = np.column_stack([xs.flatten(), ys.flatten(), np.full(64, 0.5)])
    table_track = make_track("table", [0.0, 0.0, 0.5], class_name="table")
    table_geom = make_obs_geom("table", table_pts)

    # 3 cups
    cups = []
    geometries = {"table": table_geom}
    for i in range(3):
        cid = f"cup_{i}"
        cup_pts = np.array([[i * 0.1, 0.0, 0.52], [i * 0.1 + 0.05, 0.05, 0.55]])
        cups.append(make_track(cid, [i * 0.1, 0.0, 0.53], class_name="cup"))
        geometries[cid] = make_obs_geom(cid, cup_pts)

    ref_frame = RelationReferenceFrame(
        origin_world=np.zeros(3),
        up_axis_world=np.array([0.0, 0.0, 1.0]),
        horizontal_axis_world=np.array([1.0, 0.0, 0.0]),
        depth_axis_world=np.array([0.0, 1.0, 0.0]),
    )
    ctx = FrameContext(
        frame_index=0,
        timestamp=0.0,
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=25.0, cy=25.0, width=50, height=50),
        world_T_camera=np.eye(4),
        reference_frame=ref_frame,
        camera_frame=CameraFrame.from_camera_pose(np.eye(4)),
        depth_image=np.ones((50, 50), dtype=np.float32),
        observation_geometry=geometries,
    )

    from scene_graph.geometry.plane import fit_plane_ransac as real_ransac
    with patch("scene_graph.relations.support.fit_plane_ransac", side_effect=real_ransac) as mock_ransac:
        for cup in cups:
            module.compute(cup, table_track, ctx)

        # RANSAC must have executed exactly 1 time for the table across all 3 cups!
        assert mock_ransac.call_count == 1
