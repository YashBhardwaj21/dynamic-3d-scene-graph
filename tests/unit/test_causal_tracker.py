import pytest
import numpy as np

from scene_graph.config import SceneGraphConfig
from scene_graph.perception.observation import Observation
from scene_graph.tracking.causal_tracker import CausalTracker
from scene_graph.tracking.track import TrackState


@pytest.fixture
def tracker_config():
    return SceneGraphConfig.model_validate({
        "tracking": {
            "association_threshold_m": 0.25,
            "max_missing_frames": 5,
            "min_hits_to_confirm": 3,
            "velocity_history_min": 3
        }
    })


def test_causal_tracker_lifecycle(tracker_config):
    from scene_graph.tracking.track_history import TrackHistory
    tracker = CausalTracker(tracker_config)
    tracker.history = TrackHistory()
    
    # Frame 1: New observation
    obs1 = Observation(
        obs_id="obs_1", frame_index=1, timestamp=1.0, class_name="monitor",
        confidence=0.9, bbox_xyxy=np.zeros(4), mask_rle=None,
        centroid_camera=None, centroid_world=np.array([1.0, 2.0, 3.0]),
        bbox_min_world=None, bbox_max_world=None, valid_point_count=100, point_cloud_ref=None
    )
    
    tracks = tracker.update([obs1], 1, 1.0)
    
    # Should create a candidate track
    assert len(tracks) == 1
    track = tracks[0]
    assert track.class_name == "monitor"
    assert track.state == TrackState.CANDIDATE
    assert track.observation_count == 1
    assert track.missing_count == 0
    assert np.allclose(track.centroid_world, [1.0, 2.0, 3.0])
    
    # Frame 2: Same observation (slight movement)
    obs2 = Observation(
        obs_id="obs_2", frame_index=2, timestamp=1.1, class_name="monitor",
        confidence=0.9, bbox_xyxy=np.zeros(4), mask_rle=None,
        centroid_camera=None, centroid_world=np.array([1.01, 2.0, 3.0]),
        bbox_min_world=None, bbox_max_world=None, valid_point_count=100, point_cloud_ref=None
    )
    
    tracks = tracker.update([obs2], 2, 1.1)
    assert len(tracks) == 1
    assert tracks[0].state == TrackState.CANDIDATE
    assert tracks[0].observation_count == 2
    
    # Frame 3: Same observation (confirms)
    obs3 = Observation(
        obs_id="obs_3", frame_index=3, timestamp=1.2, class_name="monitor",
        confidence=0.9, bbox_xyxy=np.zeros(4), mask_rle=None,
        centroid_camera=None, centroid_world=np.array([1.02, 2.0, 3.0]),
        bbox_min_world=None, bbox_max_world=None, valid_point_count=100, point_cloud_ref=None
    )
    
    tracks = tracker.update([obs3], 3, 1.2)
    assert len(tracks) == 1
    assert tracks[0].state == TrackState.ACTIVE
    assert tracks[0].observation_count == 3
    
    # Frame 4: Missing observation
    tracks = tracker.update([], 4, 1.3)
    assert len(tracks) == 1
    assert tracks[0].state == TrackState.TEMPORARILY_UNOBSERVED
    assert tracks[0].missing_count == 1
    
    # Frame 5-8: Continue missing, should get LOST at frame 8 (missing=5)
    tracker.update([], 5, 1.4) # missing=2
    tracker.update([], 6, 1.5) # missing=3
    tracker.update([], 7, 1.6) # missing=4
    tracks = tracker.update([], 8, 1.7) # missing=5 -> LOST
    
    assert len(tracks) == 0  # Tracker update only returns non-LOST tracks
    
    # The history should reflect all this
    assert tracker.history is not None
    transitions = tracker.history.transitions[track.object_id]
    
    # Initial: none -> candidate
    # Confirm: candidate -> active
    # Missing: active -> temporarily_unobserved
    # Lost: temporarily_unobserved -> lost
    assert len(transitions) == 4
    assert transitions[0].to_state == "candidate"
    assert transitions[1].to_state == "active"
    assert transitions[2].to_state == "temporarily_unobserved"
    assert transitions[3].to_state == "lost"
