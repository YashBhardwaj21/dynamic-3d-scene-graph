import pytest
import numpy as np

from scene_graph.tracking.track import Track, TrackState
from scene_graph.relations.context import FrameContext, ObservationGeometry
from scene_graph.relations.distance import DistanceRelationModule
from scene_graph.relations.directional import DirectionalRelationModule
from scene_graph.relations.depth_order import DepthOrderRelationModule
from scene_graph.relations.support import SupportRelationModule
from scene_graph.relations.containment import ContainmentRelationModule
from scene_graph.relations.evidence import EvidenceResult


@pytest.fixture
def dummy_context():
    from scene_graph.geometry.reference_frame import RelationReferenceFrame, CameraFrame
    ref_frame = RelationReferenceFrame.from_gravity_and_heading(
        origin_world=np.zeros(3),
        up_axis_world=np.array([0.0, 1.0, 0.0]),
        heading_world=np.array([0.0, 0.0, 1.0])
    )
    camera_frame = CameraFrame.from_camera_pose(np.eye(4))
    return FrameContext(
        frame_index=1, timestamp=1.0, intrinsics=None, world_T_camera=np.eye(4),
        reference_frame=ref_frame, camera_frame=camera_frame, depth_image=None, observation_geometry={}
    )


def create_track(obj_id, class_name, centroid):
    return Track(
        object_id=obj_id, class_name=class_name, state=TrackState.ACTIVE,
        _initial_centroid=np.array(centroid),
        last_observed_frame=1, first_observed_frame=1, observation_count=1,
        missing_count=0, detection_confidence=0.9, track_observation_ratio=1.0,
        recent_observations=[], last_timestamp=1.0
    )


def add_dummy_geometry(context, track, pts):
    """Construct an ObservationGeometry that exactly matches the dataclass schema."""
    pts_array = np.array(pts)
    context.observation_geometry[track.object_id] = ObservationGeometry(
        obs_id=f"obs_{track.object_id}_{context.frame_index}",
        track_id=track.object_id,
        centroid_world=track.centroid_world,
        bbox_min_world=np.min(pts_array, axis=0),
        bbox_max_world=np.max(pts_array, axis=0),
        depth_stats={"p05": float(np.min(pts_array[:, 2])), "median": float(np.median(pts_array[:, 2])), "p95": float(np.max(pts_array[:, 2]))},
        points_world_sampled=pts_array,
        points_world=pts_array,
        points_camera=None,
        mask=None,
        valid_point_count=len(pts_array)
    )


def test_distance_relations(dummy_context):
    module = DistanceRelationModule(near_threshold=0.40, far_threshold=1.50)
    
    t1 = create_track("t1", "cup", [0.0, 0.0, 0.0])
    t2 = create_track("t2", "book", [0.39, 0.0, 0.0])
    add_dummy_geometry(dummy_context, t1, [[0.0, 0.0, 0.0]])
    add_dummy_geometry(dummy_context, t2, [[0.39, 0.0, 0.0]])
    
    evidences = module.compute(t1, t2, dummy_context)
    # distance returns NEAR/FAR explicitly with SUPPORTED/CONTRADICTED
    preds = {e.predicate: e.result for e in evidences}
    assert preds["NEAR"] == EvidenceResult.SUPPORTED
    assert preds["FAR"] == EvidenceResult.CONTRADICTED
    
    # FAR
    t4 = create_track("t4", "book", [1.6, 0.0, 0.0])
    add_dummy_geometry(dummy_context, t4, [[1.6, 0.0, 0.0]])
    evidences = module.compute(t1, t4, dummy_context)
    preds = {e.predicate: e.result for e in evidences}
    assert preds["FAR"] == EvidenceResult.SUPPORTED
    assert preds["NEAR"] == EvidenceResult.CONTRADICTED


def test_directional_relations(dummy_context):
    module = DirectionalRelationModule(margin_x=0.05, margin_y=0.05)
    
    # LEFT_OF (dx < -0.05)
    t1 = create_track("t1", "cup", [-0.1, 0.0, 0.0])
    t2 = create_track("t2", "book", [0.0, 0.0, 0.0])
    add_dummy_geometry(dummy_context, t1, [[-0.1, 0.0, 0.0]])
    add_dummy_geometry(dummy_context, t2, [[0.0, 0.0, 0.0]])
    
    evidences = module.compute(t1, t2, dummy_context)
    preds = {e.predicate: e.result for e in evidences}
    assert preds["LEFT_OF"] == EvidenceResult.SUPPORTED
    assert preds["ABOVE"] == EvidenceResult.CONTRADICTED


def test_depth_order_relations(dummy_context):
    module = DepthOrderRelationModule(depth_margin=0.05)
    
    # IN_FRONT_OF (dz < -0.05)
    t1 = create_track("t1", "cup", [0.0, 0.0, -0.1])
    t2 = create_track("t2", "book", [0.0, 0.0, 0.0])
    add_dummy_geometry(dummy_context, t1, [[0.0, 0.0, -0.1]])
    add_dummy_geometry(dummy_context, t2, [[0.0, 0.0, 0.0]])
    
    evidences = module.compute(t1, t2, dummy_context)
    preds = {e.predicate: e.result for e in evidences}
    assert preds["IN_FRONT_OF"] == EvidenceResult.SUPPORTED

