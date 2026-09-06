import pytest
import numpy as np

from scene_graph.tracking.track import Track, TrackState
from scene_graph.relations.context import FrameContext, ObservationGeometry
from scene_graph.relations.distance import DistanceRelationModule
from scene_graph.relations.directional import DirectionalRelationModule
from scene_graph.relations.depth_order import DepthOrderRelationModule
from scene_graph.relations.support import SupportRelationModule
from scene_graph.relations.containment import ContainmentRelationModule


@pytest.fixture
def dummy_context():
    return FrameContext(
        frame_index=1, timestamp=1.0, intrinsics=None, world_T_camera=np.eye(4),
        reference_frame=None, depth_image=None, observation_geometry={}
    )


def create_track(obj_id, class_name, centroid):
    return Track(
        object_id=obj_id, class_name=class_name, state=TrackState.ACTIVE,
        centroid_world=np.array(centroid),
        last_observed_frame=1, first_observed_frame=1, observation_count=1,
        missing_count=0, detection_confidence=0.9, track_observation_ratio=1.0,
        recent_observations=[]
    )


def test_distance_relations(dummy_context):
    module = DistanceRelationModule(near_threshold=0.40, far_threshold=1.50)
    
    # NEAR
    t1 = create_track("t1", "cup", [0.0, 0.0, 0.0])
    t2 = create_track("t2", "book", [0.39, 0.0, 0.0])
    evidences = module.compute(t1, t2, dummy_context)
    assert len(evidences) == 1
    assert evidences[0].predicate == "NEAR"
    
    # NEITHER
    t3 = create_track("t3", "book", [1.0, 0.0, 0.0])
    evidences = module.compute(t1, t3, dummy_context)
    assert len(evidences) == 0
    
    # FAR
    t4 = create_track("t4", "book", [1.6, 0.0, 0.0])
    evidences = module.compute(t1, t4, dummy_context)
    assert len(evidences) == 1
    assert evidences[0].predicate == "FAR"


def test_directional_relations(dummy_context):
    module = DirectionalRelationModule(margin_x=0.05, margin_y=0.05)
    
    # LEFT_OF (dx < -0.05)
    t1 = create_track("t1", "cup", [-0.1, 0.0, 0.0])
    t2 = create_track("t2", "book", [0.0, 0.0, 0.0])
    evidences = module.compute(t1, t2, dummy_context)
    assert len(evidences) == 1
    assert evidences[0].predicate == "LEFT_OF"
    
    # ABOVE (dz > 0.05)
    t3 = create_track("t3", "cup", [0.0, 0.0, 0.1])
    evidences = module.compute(t3, t2, dummy_context)
    assert len(evidences) == 1
    assert evidences[0].predicate == "ABOVE"


def test_depth_order_relations(dummy_context):
    module = DepthOrderRelationModule(depth_margin=0.05)
    
    # IN_FRONT_OF (dy < -0.05)
    t1 = create_track("t1", "cup", [0.0, -0.1, 0.0])
    t2 = create_track("t2", "book", [0.0, 0.0, 0.0])
    evidences = module.compute(t1, t2, dummy_context)
    assert len(evidences) == 1
    assert evidences[0].predicate == "IN_FRONT_OF"
