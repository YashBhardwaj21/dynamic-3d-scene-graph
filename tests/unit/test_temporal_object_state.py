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
