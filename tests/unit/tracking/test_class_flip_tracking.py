import pytest
import numpy as np

from scene_graph.config import SceneGraphConfig
from scene_graph.tracking.causal_tracker import CausalTracker
from scene_graph.tracking.track import TrackState
from scene_graph.perception.observation import Observation
from scene_graph.geometry.point_cloud import ObjectGeometry, GeometryStatus


def _make_observation(frame_idx: int, timestamp: float, class_name: str, pos: np.ndarray, obs_id: str):
    geom = ObjectGeometry(
        status=GeometryStatus.VALID,
        centroid_world=np.asarray(pos, dtype=np.float64),
        position_covariance_world=np.eye(3, dtype=np.float64) * (0.01 ** 2),
        bbox_min_world=pos - 0.05,
        bbox_max_world=pos + 0.05,
        obb_extents_world=np.array([0.1, 0.1, 0.1]),
    )
    return Observation(
        obs_id=obs_id,
        frame_index=frame_idx,
        timestamp=timestamp,
        class_name=class_name,
        confidence=0.90,
        bbox_xyxy=np.array([100.0, 100.0, 200.0, 200.0]),
        mask_rle=None,
        object_geometry=geom,
    )


def test_label_flip_track_continuity():
    """Verify that alternating detector labels (e.g. cup -> mug -> cup) do NOT spawn new tracks."""
    config = SceneGraphConfig.from_files("configs/default.yaml")
    tracker = CausalTracker(config)

    # Frame 1: detector outputs "cup"
    obs_1 = _make_observation(
        frame_idx=1,
        timestamp=0.1,
        class_name="cup",
        pos=np.array([1.0, 0.5, 0.8]),
        obs_id="obs_001",
    )
    tracks_1 = tracker.update([obs_1], frame_index=1, timestamp=0.1)
    assert len(tracks_1) == 1
    track_id = tracks_1[0].object_id
    assert tracks_1[0].class_name == "cup"

    # Frame 2: detector flips label to "mug" at same 3D location
    obs_2 = _make_observation(
        frame_idx=2,
        timestamp=0.2,
        class_name="mug",
        pos=np.array([1.01, 0.5, 0.8]),
        obs_id="obs_002",
    )
    tracks_2 = tracker.update([obs_2], frame_index=2, timestamp=0.2)
    assert len(tracks_2) == 1
    # Physical track ID MUST survive
    assert tracks_2[0].object_id == track_id
    assert tracks_2[0].observation_count == 2
    # label_belief reflects both labels
    assert "cup" in tracks_2[0].label_belief
    assert "mug" in tracks_2[0].label_belief

    # Frame 3: detector flips back to "cup"
    obs_3 = _make_observation(
        frame_idx=3,
        timestamp=0.3,
        class_name="cup",
        pos=np.array([1.01, 0.51, 0.8]),
        obs_id="obs_003",
    )
    tracks_3 = tracker.update([obs_3], frame_index=3, timestamp=0.3)
    assert len(tracks_3) == 1
    assert tracks_3[0].object_id == track_id
    assert tracks_3[0].observation_count == 3
    assert tracks_3[0].class_name == "cup"


def test_track_label_belief_distribution():
    """Verify Track.update_label_belief tracks a normalized probability distribution."""
    from scene_graph.tracking.track import Track
    from collections import deque

    track = Track(
        object_id="track_0017",
        class_name="cup",
        state=TrackState.CANDIDATE,
        _initial_centroid=np.array([0.0, 0.0, 0.0]),
        last_observed_frame=1,
        first_observed_frame=1,
        observation_count=1,
        missing_count=0,
        detection_confidence=0.9,
        track_observation_ratio=1.0,
        last_timestamp=1.0,
        recent_observations=deque(),
    )

    assert track.label_belief == {"cup": 1.0}

    # Observe mug with confidence 0.8
    track.update_label_belief("mug", 0.8)
    assert "cup" in track.label_belief
    assert "mug" in track.label_belief
    assert np.isclose(sum(track.label_belief.values()), 1.0)
    # Cup had initial 1.0 * decay, mug added 0.8 -> cup ~0.54, mug ~0.46
    assert track.label_belief["cup"] > track.label_belief["mug"]
    assert track.primary_label == "cup"

    # Observe mug several more times -> mug becomes primary
    track.update_label_belief("mug", 0.9)
    track.update_label_belief("mug", 0.9)
    assert track.primary_label == "mug"
    assert track.class_name == "mug"
