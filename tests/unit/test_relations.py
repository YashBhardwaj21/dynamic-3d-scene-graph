import numpy as np
import pytest

from scene_graph.config import SceneGraphConfig
from scene_graph.geometry.reference_frame import (
    CameraFrame,
    RelationReferenceFrame,
)
from scene_graph.relations.context import (
    FrameContext,
    ObservationGeometry,
)
from scene_graph.relations.depth_order import DepthOrderRelationModule
from scene_graph.relations.directional import DirectionalRelationModule
from scene_graph.relations.distance import DistanceRelationModule
from scene_graph.relations.evidence import EvidenceResult
from scene_graph.tracking.track import Track, TrackState


@pytest.fixture
def config():
    return SceneGraphConfig.from_files(
        "configs/default.yaml"
    )


@pytest.fixture
def dummy_context():
    reference_frame = RelationReferenceFrame.from_gravity_and_heading(
        origin_world=np.zeros(3, dtype=np.float64),
        up_axis_world=np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
        heading_world=np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        ),
    )

    camera_frame = CameraFrame.from_camera_pose(
        np.eye(4, dtype=np.float64)
    )

    return FrameContext(
        frame_index=1,
        timestamp=1.0,
        intrinsics=None,
        world_T_camera=np.eye(4, dtype=np.float64),
        reference_frame=reference_frame,
        camera_frame=camera_frame,
        depth_image=None,
        observation_geometry={},
    )


def create_track(
    object_id: str,
    class_name: str,
    centroid,
) -> Track:
    return Track(
        object_id=object_id,
        class_name=class_name,
        state=TrackState.ACTIVE,
        _initial_centroid=np.asarray(
            centroid,
            dtype=np.float64,
        ),
        last_observed_frame=1,
        first_observed_frame=1,
        observation_count=1,
        missing_count=0,
        detection_confidence=0.9,
        track_observation_ratio=1.0,
        recent_observations=[],
        last_timestamp=1.0,
    )


def add_dummy_geometry(
    context: FrameContext,
    track: Track,
    points,
) -> None:
    points_array = np.asarray(
        points,
        dtype=np.float64,
    )

    context.observation_geometry[track.object_id] = (
        ObservationGeometry(
            obs_id=(
                f"obs_{track.object_id}_"
                f"{context.frame_index}"
            ),
            track_id=track.object_id,
            centroid_world=track.centroid_world,
            bbox_min_world=np.min(
                points_array,
                axis=0,
            ),
            bbox_max_world=np.max(
                points_array,
                axis=0,
            ),
            depth_stats={
                "p05": float(
                    np.percentile(
                        points_array[:, 2],
                        5,
                    )
                ),
                "median": float(
                    np.median(
                        points_array[:, 2]
                    )
                ),
                "p95": float(
                    np.percentile(
                        points_array[:, 2],
                        95,
                    )
                ),
            },
            points_world_sampled=points_array,
            points_world=points_array,
            points_camera=None,
            mask=None,
            valid_point_count=len(points_array),
        )
    )


def test_distance_relations(
    config,
    dummy_context,
):
    module = DistanceRelationModule(config)

    near_threshold = config.relations.distance.near_threshold
    far_threshold = config.relations.distance.far_threshold

    near_distance = near_threshold * 0.5
    far_distance = far_threshold * 1.1

    t1 = create_track(
        "t1",
        "cup",
        [0.0, 0.0, 0.0],
    )

    t2 = create_track(
        "t2",
        "book",
        [near_distance, 0.0, 0.0],
    )

    add_dummy_geometry(
        dummy_context,
        t1,
        [[0.0, 0.0, 0.0]],
    )

    add_dummy_geometry(
        dummy_context,
        t2,
        [[near_distance, 0.0, 0.0]],
    )

    evidences = module.compute(
        t1,
        t2,
        dummy_context,
    )

    assert len(evidences) == 1
    assert evidences[0].predicate == "NEAR"
    assert evidences[0].result == EvidenceResult.SUPPORTED

    t3 = create_track(
        "t3",
        "book",
        [far_distance, 0.0, 0.0],
    )

    add_dummy_geometry(
        dummy_context,
        t3,
        [[far_distance, 0.0, 0.0]],
    )

    evidences = module.compute(
        t1,
        t3,
        dummy_context,
    )

    assert len(evidences) == 1
    assert evidences[0].predicate == "FAR"
    assert evidences[0].result == EvidenceResult.SUPPORTED


def test_directional_relations(
    config,
    dummy_context,
):
    module = DirectionalRelationModule(config)

    margin_x = config.relations.directional.margin_x
    margin_y = config.relations.directional.margin_y

    x_offset = margin_x * 2.0
    y_offset = margin_y * 2.0

    t1 = create_track(
        "t1",
        "cup",
        [-x_offset, 0.0, 0.0],
    )

    t2 = create_track(
        "t2",
        "book",
        [0.0, 0.0, 0.0],
    )

    add_dummy_geometry(
        dummy_context,
        t1,
        [[-x_offset, 0.0, 0.0]],
    )

    add_dummy_geometry(
        dummy_context,
        t2,
        [[0.0, 0.0, 0.0]],
    )

    evidences = module.compute(
        t1,
        t2,
        dummy_context,
    )

    predicates = {
        evidence.predicate: evidence.result
        for evidence in evidences
    }

    assert predicates["LEFT_OF"] == EvidenceResult.SUPPORTED

    t3 = create_track(
        "t3",
        "book",
        [0.0, y_offset, 0.0],
    )

    add_dummy_geometry(
        dummy_context,
        t3,
        [[0.0, y_offset, 0.0]],
    )

    evidences = module.compute(
        t1,
        t3,
        dummy_context,
    )

    predicates = {
        evidence.predicate: evidence.result
        for evidence in evidences
    }

    assert predicates["ABOVE"] == EvidenceResult.CONTRADICTED


def test_depth_order_relations(
    config,
    dummy_context,
):
    module = DepthOrderRelationModule(config)

    depth_margin = (
        config.relations.depth_order.depth_margin
    )

    depth_offset = depth_margin * 2.0

    t1 = create_track(
        "t1",
        "cup",
        [0.0, 0.0, -depth_offset],
    )

    t2 = create_track(
        "t2",
        "book",
        [0.0, 0.0, 0.0],
    )

    add_dummy_geometry(
        dummy_context,
        t1,
        [[0.0, 0.0, -depth_offset]],
    )

    add_dummy_geometry(
        dummy_context,
        t2,
        [[0.0, 0.0, 0.0]],
    )

    evidences = module.compute(
        t1,
        t2,
        dummy_context,
    )

    predicates = {
        evidence.predicate: evidence.result
        for evidence in evidences
    }

    assert predicates["IN_FRONT_OF"] == EvidenceResult.SUPPORTED