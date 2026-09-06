import pytest
from scene_graph.tracking.track import Track, TrackState
from scene_graph.temporal.object_state import ObjectState, ObjectStateMachine


def create_track(obj_id, state):
    # Dummy track creation for tests
    return Track(
        object_id=obj_id, class_name="monitor", state=state,
        centroid_world=None, last_observed_frame=1, first_observed_frame=1,
        observation_count=1, missing_count=0, detection_confidence=0.9,
        track_observation_ratio=1.0, recent_observations=[], velocity_world=None,
        last_timestamp=1.0
    )


def test_object_state_machine():
    machine = ObjectStateMachine(hysteresis_frames=2)
    
    # 1. New track comes in as ACTIVE -> STABLE
    t1 = create_track("obj_1", TrackState.ACTIVE)
    states = machine.update([t1], frame_index=1)
    
    assert "obj_1" in states
    assert states["obj_1"] == ObjectState.STABLE
    
    # 2. Track goes TEMPORARILY_UNOBSERVED
    t1.state = TrackState.TEMPORARILY_UNOBSERVED
    states = machine.update([t1], frame_index=2)
    # hysteresis = 2, so 1 frame of missing should stay STABLE
    assert states["obj_1"] == ObjectState.STABLE
    
    # 3. Missing again
    states = machine.update([t1], frame_index=3)
    assert states["obj_1"] == ObjectState.STABLE
    
    # 4. Missing again (exceeds hysteresis)
    states = machine.update([t1], frame_index=4)
    assert states["obj_1"] == ObjectState.UNSTABLE
    
    # 5. Comes back ACTIVE
    t1.state = TrackState.ACTIVE
    states = machine.update([t1], frame_index=5)
    # Doesn't immediately become STABLE, takes hysteresis
    assert states["obj_1"] == ObjectState.UNSTABLE
    
    states = machine.update([t1], frame_index=6)
    assert states["obj_1"] == ObjectState.STABLE
    
    # 6. Goes LOST
    t1.state = TrackState.LOST
    states = machine.update([t1], frame_index=7)
    assert states["obj_1"] == ObjectState.UNKNOWN
