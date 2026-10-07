from collections import deque
import numpy as np
import pytest

from scene_graph.config import SceneGraphConfig
from scene_graph.relations.admissibility import AdmissibilityFilter
from scene_graph.relations.context import FrameContext, ObservationGeometry
from scene_graph.geometry.reference_frame import RelationReferenceFrame, CameraFrame
from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.relations.evidence import ReferenceFrameType
from scene_graph.tracking.track import Track, TrackState


def make_track(
    object_id: str,
    class_name: str,
    centroid: np.ndarray,
    size: np.ndarray,
) -> Track:
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
        size_world=np.asarray(size, dtype=np.float64),
    )


def test_unknown_planar_surface_supports_cup():
    """Asserts an unclassified planar surface ('unknown_slab') is geometrically admissible to support a cup."""
    config = SceneGraphConfig.from_files("configs/default.yaml")
    filter_ = AdmissibilityFilter(config)

    # Slab at z=0.5 with thickness 0.1, size (1.0, 1.0, 0.1) -> top at z=0.55
    slab = make_track("slab_1", "unknown_slab", centroid=[0.0, 0.0, 0.5], size=[1.0, 1.0, 0.1])
    # Cup at z=0.62 with height 0.14 -> bottom at z=0.55
    cup = make_track("cup_1", "cup", centroid=[0.0, 0.0, 0.62], size=[0.1, 0.1, 0.14])

    assert filter_.is_admissible("ON", cup, slab) is True
    # Inverse: slab is UNDER cup
    assert filter_.is_admissible("UNDER", slab, cup) is True


def test_unlikely_geometry_rejected():
    """Asserts object B placed physically above object A cannot be supported by A (ON(A, B) is False)."""
    config = SceneGraphConfig.from_files("configs/default.yaml")
    filter_ = AdmissibilityFilter(config)

    # Object A is lower (z=0.2)
    track_a = make_track("obj_A", "cup", centroid=[0.0, 0.0, 0.2], size=[0.1, 0.1, 0.1])
    # Object B is higher (z=0.8)
    track_b = make_track("obj_B", "table", centroid=[0.0, 0.0, 0.8], size=[1.0, 1.0, 0.1])

    # Asking if A is ON B when A is below B -> False!
    assert filter_.is_admissible("ON", track_a, track_b) is False


def test_non_overlapping_horizontal_rejected():
    """Asserts objects with no horizontal overlap cannot satisfy ON relation."""
    config = SceneGraphConfig.from_files("configs/default.yaml")
    filter_ = AdmissibilityFilter(config)

    # Slab at origin
    slab = make_track("slab_1", "desk", centroid=[0.0, 0.0, 0.5], size=[0.5, 0.5, 0.1])
    # Cup 2 meters to the side (x=2.0)
    cup = make_track("cup_1", "cup", centroid=[2.0, 0.0, 0.6], size=[0.1, 0.1, 0.1])

    assert filter_.is_admissible("ON", cup, slab) is False


def test_open_vocabulary_purely_geometric():
    """Asserts unknown or novel class names are evaluated purely on physical geometry."""
    config = SceneGraphConfig.from_files("configs/default.yaml")
    filter_ = AdmissibilityFilter(config)

    # Completely novel labels not in any pre-defined category
    novel_surface = make_track("surf_1", "robot_charging_pad", centroid=[0.0, 0.0, 0.5], size=[1.0, 1.0, 0.1])
    novel_object = make_track("obj_1", "sensor_dongle", centroid=[0.0, 0.0, 0.62], size=[0.1, 0.1, 0.14])

    assert filter_.is_admissible("ON", novel_object, novel_surface) is True
    assert filter_.is_admissible("UNDER", novel_surface, novel_object) is True


def test_missing_geometry_rejected():
    """Asserts that missing geometry causes ON admissibility to return False."""
    config = SceneGraphConfig.from_files("configs/default.yaml")
    filter_ = AdmissibilityFilter(config)

    track_no_geom = Track(
        object_id="no_geom_1",
        class_name="cup",
        state=TrackState.ACTIVE,
        _initial_centroid=None,
        last_observed_frame=0,
        first_observed_frame=0,
        observation_count=1,
        missing_count=0,
        detection_confidence=0.9,
        track_observation_ratio=1.0,
        last_timestamp=0.0,
        recent_observations=deque(),
        size_world=None,
    )
    table = make_track("table_1", "table", centroid=[0.0, 0.0, 0.5], size=[1.0, 1.0, 0.1])

    assert filter_.is_admissible("ON", track_no_geom, table) is False
    assert filter_.is_admissible("ON", table, track_no_geom) is False
