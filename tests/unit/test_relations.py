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


def test_candidate_pairs_lifecycle_boundary(config):
    """Verify RelationRegistry candidate pairs include ACTIVE and TEMPORARILY_UNOBSERVED, but exclude LOST."""
    from scene_graph.relations.registry import RelationRegistry

    registry = RelationRegistry(config)

    t_active = create_track("t_active", "cup", [0.0, 0.0, 0.0])
    t_active.state = TrackState.ACTIVE

    t_unobserved = create_track("t_unobs", "book", [1.0, 0.0, 0.0])
    t_unobserved.state = TrackState.TEMPORARILY_UNOBSERVED

    t_lost = create_track("t_lost", "bottle", [2.0, 0.0, 0.0])
    t_lost.state = TrackState.LOST

    pairs = registry._generate_candidate_pairs([t_active, t_unobserved, t_lost])

    # Should contain pairs between t_active and t_unobs only (2 pairs total)
    assert len(pairs) == 2
    track_ids_in_pairs = {(p[0].object_id, p[1].object_id) for p in pairs}
    assert ("t_active", "t_unobs") in track_ids_in_pairs
    assert ("t_unobs", "t_active") in track_ids_in_pairs
    assert not any("t_lost" in (p[0].object_id, p[1].object_id) for p in pairs)


def test_provenance_confidence_adjustment(config, dummy_context):
    """Verify that predicted geometry dampens relation evidence confidence."""
    from scene_graph.relations.registry import RelationRegistry
    from scene_graph.geometry.provenance import GeometrySource
    from scene_graph.relations.evidence import RelationEvidence, EvidenceResult, ReferenceFrameType

    registry = RelationRegistry(config)
    scale = config.relations.predicted_geometry_confidence_scale  # 0.5 by default

    t1 = create_track("t1", "cup", [0.0, 0.0, 0.0])
    t2 = create_track("t2", "book", [0.1, 0.0, 0.0])

    add_dummy_geometry(dummy_context, t1, [[0.0, 0.0, 0.0]])
    add_dummy_geometry(dummy_context, t2, [[0.1, 0.0, 0.0]])

    # Case 1: Both OBSERVED
    dummy_context.observation_geometry["t1"].geometry_source = GeometrySource.OBSERVED
    dummy_context.observation_geometry["t2"].geometry_source = GeometrySource.OBSERVED

    base_evidence = RelationEvidence(
        predicate="NEAR",
        subject_id="t1",
        object_id="t2",
        frame_index=1,
        timestamp=1.0,
        result=EvidenceResult.SUPPORTED,
        value=0.1,
        threshold=0.5,
        confidence=0.8,
        reference_frame=ReferenceFrameType.WORLD,
        evidence_type="direct",
        details={},
    )

    ev_obs = registry._adjust_evidence_for_geometry_provenance(base_evidence, dummy_context)
    assert ev_obs.confidence == pytest.approx(0.8)
    assert "geometry_provenance" not in ev_obs.details

    # Case 2: One PREDICTED
    dummy_context.observation_geometry["t1"].geometry_source = GeometrySource.PREDICTED
    ev_pred1 = registry._adjust_evidence_for_geometry_provenance(base_evidence, dummy_context)
    assert ev_pred1.confidence == pytest.approx(0.8 * scale)
    assert ev_pred1.details["geometry_provenance"]["predicted_endpoint_count"] == 1

    # Case 3: Both PREDICTED
    dummy_context.observation_geometry["t2"].geometry_source = GeometrySource.PREDICTED
    ev_pred2 = registry._adjust_evidence_for_geometry_provenance(base_evidence, dummy_context)
    assert ev_pred2.confidence == pytest.approx(0.8 * (scale ** 2))
    assert ev_pred2.details["geometry_provenance"]["predicted_endpoint_count"] == 2