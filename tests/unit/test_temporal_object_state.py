import pytest
from scene_graph.tracking.track import Track, TrackState
from scene_graph.temporal.object_state import ObjectState, ObjectStateMachine


def create_track(obj_id, state):
    # Dummy track creation for tests — uses current Track schema
    import numpy as np
    return Track(
        object_id=obj_id, class_name="monitor", state=state,
        _initial_centroid=np.zeros(3), last_observed_frame=1, first_observed_frame=1,
        observation_count=1, missing_count=0, detection_confidence=0.9,
        track_observation_ratio=1.0, recent_observations=[],
        last_timestamp=1.0
    )


def test_object_state_machine():
    # ObjectStateMachine no longer takes hysteresis_frames — it maps
    # TrackState directly to ObjectState without frame-counting.
    machine = ObjectStateMachine()
    
    # 1. New track comes in as ACTIVE -> STABLE
    t1 = create_track("obj_1", TrackState.ACTIVE)
    states = machine.update([t1], frame_index=1)
    
    assert "obj_1" in states
    assert states["obj_1"] == ObjectState.STABLE
    
    # 2. Track goes TEMPORARILY_UNOBSERVED -> UNSTABLE
    t1.state = TrackState.TEMPORARILY_UNOBSERVED
    states = machine.update([t1], frame_index=2)
    assert states["obj_1"] == ObjectState.UNSTABLE
    
    # 3. Comes back ACTIVE -> STABLE again
    t1.state = TrackState.ACTIVE
    states = machine.update([t1], frame_index=3)
    assert states["obj_1"] == ObjectState.STABLE
    
    # 4. Goes LOST -> UNKNOWN
    t1.state = TrackState.LOST
    states = machine.update([t1], frame_index=4)
    assert states["obj_1"] == ObjectState.UNKNOWN


def test_unified_graph_lifecycle_mapping():
    """Verify explicit mapping from tracker state to graph representation and visibility."""
    from scene_graph.graph.temporal_graph import TemporalSceneGraph
    from scene_graph.graph.participation_state import GraphParticipationState

    machine = ObjectStateMachine()
    graph = TemporalSceneGraph()

    # 1. CANDIDATE -> not added to graph
    t_cand = create_track("cand_1", TrackState.CANDIDATE)
    states = machine.update([t_cand], frame_index=1)
    graph.update_nodes([t_cand], states)
    assert "cand_1" not in graph.nodes

    # 2. ACTIVE -> added to graph, active, observed
    t_act = create_track("act_1", TrackState.ACTIVE)
    states = machine.update([t_act], frame_index=2)
    graph.update_nodes([t_act], states)
    assert "act_1" in graph.nodes
    node_act = graph.nodes["act_1"]
    assert node_act.participation == GraphParticipationState.ACTIVE
    assert node_act.is_active is True
    assert node_act.is_observed is True
    assert node_act.attributes["observed"] is True
    assert node_act.attributes["visibility"] == "observed"

    # 3. TEMPORARILY_UNOBSERVED -> remains active in graph, predicted/unobserved
    t_act.state = TrackState.TEMPORARILY_UNOBSERVED
    states = machine.update([t_act], frame_index=3)
    graph.update_nodes([t_act], states)
    assert "act_1" in graph.nodes
    assert node_act.participation == GraphParticipationState.ACTIVE
    assert node_act.is_active is True
    assert node_act.is_observed is False
    assert node_act.attributes["observed"] is False
    assert node_act.attributes["visibility"] == "predicted"

    # 4. LOST -> retired to REMOVED
    t_act.state = TrackState.LOST
    states = machine.update([t_act], frame_index=4)
    graph.update_nodes([t_act], states)
    assert "act_1" in graph.nodes
    assert node_act.participation == GraphParticipationState.REMOVED
    assert node_act.is_active is False

