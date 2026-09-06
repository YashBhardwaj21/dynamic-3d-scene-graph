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
            "confirmation": {"min_hits": 3},
            "occlusion": {"max_missing_seconds": 0.5},
            "association": {"max_distance_m": 0.25},
        }
    })


def _make_obs(obs_id, frame_index, timestamp, centroid):
    """Helper to create a minimal Observation with the current dataclass schema."""
    return Observation(
        obs_id=obs_id,
        frame_index=frame_index,
        timestamp=timestamp,
        class_name="monitor",
        confidence=0.9,
        bbox_xyxy=np.zeros(4),
        mask_rle=None,
    )


def test_causal_tracker_lifecycle(tracker_config):
    tracker = CausalTracker(tracker_config)

    # We need to feed centroids through the tracker's own geometry path.
    # The tracker associates observations by centroid, so we attach object_geometry.
    from scene_graph.geometry.point_cloud import ObjectGeometry

    def make_obs_with_geo(obs_id, frame_index, timestamp, centroid):
        obs = _make_obs(obs_id, frame_index, timestamp, centroid)
        obs.object_geometry = ObjectGeometry(
            centroid_world=np.array(centroid),
            centroid_camera=np.array(centroid),
            bbox_min_world=np.array(centroid) - 0.01,
            bbox_max_world=np.array(centroid) + 0.01,
            valid_point_count=100,
        )
        return obs

    # Frame 1: New observation
    obs1 = make_obs_with_geo("obs_1", 1, 1.0, [1.0, 2.0, 3.0])
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
    obs2 = make_obs_with_geo("obs_2", 2, 1.1, [1.01, 2.0, 3.0])
    tracks = tracker.update([obs2], 2, 1.1)
    assert len(tracks) == 1
    assert tracks[0].state == TrackState.CANDIDATE
    assert tracks[0].observation_count == 2

    # Frame 3: Same observation (confirms after min_hits=3)
    obs3 = make_obs_with_geo("obs_3", 3, 1.2, [1.02, 2.0, 3.0])
    tracks = tracker.update([obs3], 3, 1.2)
    assert len(tracks) == 1
    assert tracks[0].state == TrackState.ACTIVE
    assert tracks[0].observation_count == 3

    # Frame 4: Missing observation
    tracks = tracker.update([], 4, 1.3)
    assert len(tracks) == 1
    assert tracks[0].state == TrackState.TEMPORARILY_UNOBSERVED
    assert tracks[0].missing_count == 1

    # Frames 5-8: Continue missing. With max_missing_seconds=0.5 and dt=0.1s per frame,
    # the track should become LOST once elapsed missing time > 0.5s.
    # Frame 5 @ 1.4s (missing 0.2s), Frame 6 @ 1.5s (0.3s), Frame 7 @ 1.6s (0.4s),
    # Frame 8 @ 1.7s (0.5s), Frame 9 @ 1.8s (0.6s > 0.5s) -> LOST
    tracker.update([], 5, 1.4)
    tracker.update([], 6, 1.5)
    tracker.update([], 7, 1.6)
    tracker.update([], 8, 1.7)
    tracks = tracker.update([], 9, 1.8)

    # Tracker should have dropped the LOST track from the returned list
    lost_tracks = [t for t in tracks if t.state == TrackState.LOST]
    active_or_temp = [t for t in tracks if t.state in (TrackState.ACTIVE, TrackState.TEMPORARILY_UNOBSERVED)]
    # Either the tracker removed the track entirely or marked it LOST
    assert len(active_or_temp) == 0
