"""Unit tests for offline semantic mapping, occupancy grid, and global association."""

import math
from pathlib import Path
import numpy as np
import pytest

from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.transforms import pose_to_transform, transform_points
from scene_graph.geometry.plane import fit_plane_ransac
from scene_graph.mapping.occupancy_grid import OccupancyGrid, OccupancyGridConfig
from scene_graph.mapping.global_associator import (
    GlobalObjectAssociator,
    ObjectObservationRecord,
)


def test_camera_projection_and_transform():
    """Verify camera unprojection and SE(3) transformation contract."""
    intrinsics = CameraIntrinsics(
        fx=384.0, fy=384.0, cx=320.0, cy=240.0,
        width=640, height=480
    )

    # Pixel at optical center (320, 240) at depth 2.0m
    u = np.array([320.0])
    v = np.array([240.0])
    z = np.array([2.0])

    pts_cam = intrinsics.pixels_to_camera(u, v, z)
    assert np.allclose(pts_cam[0], [0.0, 0.0, 2.0], atol=1e-5)

    # Identity pose
    T_wc = pose_to_transform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    pts_world = transform_points(T_wc, pts_cam)
    assert np.allclose(pts_world[0], [0.0, 0.0, 2.0], atol=1e-5)

    # Translated pose: tx=1.0, ty=2.0, tz=3.0
    T_wc_trans = pose_to_transform(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)
    pts_world_trans = transform_points(T_wc_trans, pts_cam)
    assert np.allclose(pts_world_trans[0], [1.0, 2.0, 5.0], atol=1e-5)


def test_global_object_associator_persistent_ids():
    """Verify persistent global IDs, one-to-one matching, and candidate promotion."""
    associator = GlobalObjectAssociator(
        association_dist_threshold_m=0.50,
        min_hits_to_confirm=2,
        voxel_size_m=0.01,
    )

    # Frame 0: Two chairs at (0, 0, 2.0) and (1.5, 0, 2.0)
    obs1 = ObjectObservationRecord(
        obs_id="obs_00_01",
        frame_index=0,
        timestamp=100.0,
        class_name="chair",
        confidence=0.85,
        bbox_xyxy=[100, 100, 200, 200],
        centroid_camera=[0.0, 0.0, 2.0],
        centroid_world=[0.0, 0.0, 2.0],
        points_world=np.array([[0.0, 0.0, 2.0], [0.05, 0.0, 2.0]]),
    )
    obs2 = ObjectObservationRecord(
        obs_id="obs_00_02",
        frame_index=0,
        timestamp=100.0,
        class_name="chair",
        confidence=0.75,
        bbox_xyxy=[300, 100, 400, 200],
        centroid_camera=[1.5, 0.0, 2.0],
        centroid_world=[1.5, 0.0, 2.0],
        points_world=np.array([[1.5, 0.0, 2.0], [1.55, 0.0, 2.0]]),
    )

    mapping0 = associator.update([obs1, obs2])
    assert "obs_00_01" in mapping0
    assert "obs_00_02" in mapping0
    id1 = mapping0["obs_00_01"]
    id2 = mapping0["obs_00_02"]
    assert id1 != id2
    assert id1 == "chair_001"
    assert id2 == "chair_002"

    # Frame 1: Re-observe the first chair near (0.05, 0, 2.02)
    obs3 = ObjectObservationRecord(
        obs_id="obs_01_01",
        frame_index=1,
        timestamp=100.1,
        class_name="chair",
        confidence=0.90,
        bbox_xyxy=[105, 102, 202, 201],
        centroid_camera=[0.05, 0.0, 2.02],
        centroid_world=[0.05, 0.0, 2.02],
        points_world=np.array([[0.05, 0.0, 2.02], [0.06, 0.0, 2.02]]),
    )
    mapping1 = associator.update([obs3])
    assert mapping1["obs_01_01"] == id1  # Persistent ID preserved!

    # Since min_hits_to_confirm=2, chair_001 should now be confirmed
    assert id1 in associator.entities
    assert associator.entities[id1].is_confirmed is True


def test_geometry_fusion_and_footprint():
    """Verify point cloud fusion, percentile bounding box, and 2D footprint."""
    associator = GlobalObjectAssociator(
        association_dist_threshold_m=0.50,
        min_hits_to_confirm=1,
        voxel_size_m=0.01,
    )

    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.0, 0.5],
        [0.2, -0.5, 0.2],
    ])
    obs = ObjectObservationRecord(
        obs_id="obs_01",
        frame_index=0,
        timestamp=10.0,
        class_name="table",
        confidence=0.9,
        bbox_xyxy=[50, 50, 150, 150],
        centroid_camera=[0.25, -0.1, 0.25],
        centroid_world=[0.25, -0.1, 0.25],
        points_world=pts,
    )
    associator.update([obs])
    associator.finalize_geometries(floor_normal=np.array([0.0, -1.0, 0.0]), floor_d=0.5)

    entity = associator.entities["table_001"]
    assert entity.dimensions is not None
    assert len(entity.floor_footprint_polygon) >= 3


def test_occupancy_grid_raycasting_and_inflation(tmp_path: Path):
    """Verify free-space clearing, obstacle marking, and obstacle inflation."""
    cfg = OccupancyGridConfig(
        resolution_m=0.10,
        origin_x_m=-2.0,
        origin_y_m=-2.0,
        width_m=4.0,
        height_m=4.0,
        min_obstacle_height_m=0.1,
        max_obstacle_height_m=1.5,
        floor_band_m=0.05,
        robot_radius_m=0.20,
        safety_margin_m=0.00,
    )
    grid = OccupancyGrid(cfg)

    # Initial state: all unknown (-1)
    assert np.all(grid.grid == -1)

    # Camera at (0, 0). Point at (1.0, 0.0, 0.5) is an obstacle (height 0.5m)
    pts = np.array([[1.0, 0.0, 0.5]])
    grid.integrate_depth_points(camera_floor_x=0.0, camera_floor_y=0.0, points_floor_aligned=pts)

    # Cell at (1.0, 0.0) should be OCCUPIED
    obs_c, obs_r = grid.world_to_grid(1.0, 0.0)
    assert grid.grid[obs_r, obs_c] == OccupancyGrid.OCCUPIED

    # Ray intermediate cells (e.g. 0.5, 0.0) should be FREE
    mid_c, mid_r = grid.world_to_grid(0.5, 0.0)
    assert grid.grid[mid_r, mid_c] == OccupancyGrid.FREE

    # Test inflation: cells within 0.20m of (1.0, 0.0) should become OCCUPIED in inflated grid
    inflated = grid.get_inflated_grid()
    adj_c, adj_r = grid.world_to_grid(0.9, 0.0)
    assert inflated[adj_r, adj_c] == OccupancyGrid.OCCUPIED

    # Export ROS map
    pgm_path = tmp_path / "map.pgm"
    yaml_path = tmp_path / "map.yaml"
    grid.export_ros_map(pgm_path, yaml_path)
    assert pgm_path.exists()
    assert yaml_path.exists()
