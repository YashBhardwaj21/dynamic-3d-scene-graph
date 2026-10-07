"""Unit tests for Stage 7: Temporal Tracking & Association Robustness."""

import numpy as np
import pytest

from scene_graph.config import SceneGraphConfig
from scene_graph.geometry.point_cloud import ObjectGeometry, GeometryStatus
from scene_graph.perception.observation import Observation
from scene_graph.tracking.causal_tracker import CausalTracker
from scene_graph.tracking.track import TrackState


def _create_obs(
    obs_id: str,
    frame_index: int,
    timestamp: float,
    pos: list[float] | np.ndarray,
    class_name: str = "cup",
    cov_std: float = 0.01,
) -> Observation:
    pos_arr = np.asarray(pos, dtype=np.float64)
    geom = ObjectGeometry(
        status=GeometryStatus.VALID,
        centroid_world=pos_arr.copy(),
        centroid_camera=pos_arr.copy(),
        position_covariance_world=np.eye(3, dtype=np.float64) * (cov_std ** 2),
        bbox_min_world=pos_arr - 0.05,
        bbox_max_world=pos_arr + 0.05,
        obb_extents_world=np.array([0.1, 0.1, 0.1]),
        valid_point_count=50,
    )
    return Observation(
        obs_id=obs_id,
        frame_index=frame_index,
        timestamp=timestamp,
        class_name=class_name,
        confidence=0.92,
        bbox_xyxy=np.array([100.0, 100.0, 200.0, 200.0]),
        mask_rle=None,
        object_geometry=geom,
    )


def test_unobserved_covariance_growth():
    """Verify that Kalman position covariance grows strictly as an object remains unobserved."""
    config = SceneGraphConfig.from_files("configs/default.yaml")
    config.tracking.confirmation.min_hits = 1
    config.tracking.occlusion.max_missing_seconds = 2.0
    tracker = CausalTracker(config)

    # Frame 1: Initial observation at (1.0, 0.0, 0.0)
    obs = _create_obs("obs_1", 1, 1.0, [1.0, 0.0, 0.0], cov_std=0.02)
    tracks = tracker.update([obs], 1, 1.0)
    assert len(tracks) == 1
    init_cov = tracks[0].position_covariance_world[0, 0]

    # Frame 2: Missing observation (elapsed 0.2s)
    tracks = tracker.update([], 2, 1.2)
    assert len(tracks) == 1
    cov_frame2 = tracks[0].position_covariance_world[0, 0]
    assert cov_frame2 > init_cov

    # Frame 3: Missing observation (elapsed 0.5s total)
    tracks = tracker.update([], 3, 1.5)
    assert len(tracks) == 1
    cov_frame3 = tracks[0].position_covariance_world[0, 0]
    assert cov_frame3 > cov_frame2


def test_velocity_extrapolation_during_occlusion():
    """Verify that a moving object's position extrapolates forward smoothly while occluded."""
    config = SceneGraphConfig.from_files("configs/default.yaml")
    config.tracking.confirmation.min_hits = 2
    config.tracking.occlusion.max_missing_seconds = 2.0
    config.tracking.association.max_distance_m = 1.0
    tracker = CausalTracker(config)

    # Object moving along X at 1.0 m/s
    # Frame 1 @ t=1.0: x=0.0
    obs1 = _create_obs("obs_1", 1, 1.0, [0.0, 0.0, 1.0])
    tracker.update([obs1], 1, 1.0)

    # Frame 2 @ t=1.2: x=0.2 (v ~ 1.0 m/s)
    obs2 = _create_obs("obs_2", 2, 1.2, [0.2, 0.0, 1.0])
    tracks = tracker.update([obs2], 2, 1.2)
    assert len(tracks) == 1
    track_id = tracks[0].object_id
    assert tracks[0].state == TrackState.ACTIVE

    # Frame 3 @ t=1.4: Occluded / Missing observation
    tracks = tracker.update([], 3, 1.4)
    assert len(tracks) == 1
    # Position should have advanced forward along X from 0.2
    assert tracks[0].centroid_world[0] > 0.2
    assert tracks[0].state == TrackState.TEMPORARILY_UNOBSERVED

    # Frame 4 @ t=1.6: Still occluded
    tracks = tracker.update([], 4, 1.6)
    assert len(tracks) == 1
    assert tracks[0].centroid_world[0] > 0.3

    # Frame 5 @ t=1.8: Object reappears at x=0.8 (matching the motion trajectory)
    obs5 = _create_obs("obs_5", 5, 1.8, [0.8, 0.0, 1.0])
    tracks = tracker.update([obs5], 5, 1.8)

    # Must associate to the same track without spawning a new one
    assert len(tracks) == 1
    assert tracks[0].object_id == track_id
    assert tracks[0].state == TrackState.ACTIVE
    assert tracks[0].observation_count == 3


def test_candidate_missing_lifecycle_preserves_min_hits():
    """Verify candidate tracks do not prematurely promote to ACTIVE after missing frames."""
    config = SceneGraphConfig.from_files("configs/default.yaml")
    config.tracking.confirmation.min_hits = 3
    config.tracking.occlusion.max_missing_seconds = 1.0
    tracker = CausalTracker(config)

    # Frame 1: Hit 1 -> CANDIDATE
    obs1 = _create_obs("obs_1", 1, 1.0, [1.0, 1.0, 1.0])
    tracks = tracker.update([obs1], 1, 1.0)
    assert tracks[0].state == TrackState.CANDIDATE
    assert tracks[0].observation_count == 1

    # Frame 2: Missing -> TEMPORARILY_UNOBSERVED
    tracks = tracker.update([], 2, 1.1)
    assert tracks[0].state == TrackState.TEMPORARILY_UNOBSERVED
    assert tracks[0].observation_count == 1

    # Frame 3: Hit 2 -> Must return to CANDIDATE (not ACTIVE, since hits=2 < min_hits=3)
    obs3 = _create_obs("obs_3", 3, 1.2, [1.01, 1.0, 1.0])
    tracks = tracker.update([obs3], 3, 1.2)
    assert tracks[0].state == TrackState.CANDIDATE
    assert tracks[0].observation_count == 2

    # Frame 4: Hit 3 -> Now ACTIVE
    obs4 = _create_obs("obs_4", 4, 1.3, [1.02, 1.0, 1.0])
    tracks = tracker.update([obs4], 4, 1.3)
    assert tracks[0].state == TrackState.ACTIVE
    assert tracks[0].observation_count == 3


def test_mahalanobis_gating_rejects_statistical_outliers():
    """Verify that observations outside the chi-square covariance gate are rejected."""
    config = SceneGraphConfig.from_files("configs/default.yaml")
    config.tracking.association.max_distance_m = 1.0
    # Set chi2_probability so threshold is strict
    config.tracking.gating.chi2_probability = 0.95
    tracker = CausalTracker(config)

    # Initialize track with small uncertainty (cov = 0.005^2)
    obs1 = _create_obs("obs_1", 1, 1.0, [0.0, 0.0, 0.0], cov_std=0.005)
    tracks = tracker.update([obs1], 1, 1.0)
    assert len(tracks) == 1

    # Observation at 0.75m: well within max_distance_m (1.0m),
    # but with predicted sigma ~ 0.10m, mahalanobis_sq ~ 56 >> chi2_threshold (~7.8)
    obs_outlier = _create_obs("obs_outlier", 2, 1.1, [0.75, 0.0, 0.0], cov_std=0.005)
    tracks = tracker.update([obs_outlier], 2, 1.1)

    # The outlier must NOT match the original track; a second track is created
    assert len(tracks) == 2
    track_ids = [t.object_id for t in tracks]
    assert len(set(track_ids)) == 2


def test_delayed_observation_replay_with_unobserved_frames():
    """Verify that update_delayed properly rolls forward through frames with unobserved intervals."""
    config = SceneGraphConfig.from_files("configs/default.yaml")
    config.tracking.confirmation.min_hits = 1
    config.tracking.occlusion.max_missing_seconds = 2.0
    tracker = CausalTracker(config)

    # Frame 1 @ t=1.0: cup observed
    obs1 = _create_obs("c1", 1, 1.0, [1.0, 0.0, 0.0], class_name="cup")
    tracker.update([obs1], 1, 1.0)

    # Frame 2 @ t=1.1: unobserved
    tracker.update([], 2, 1.1)

    # Frame 3 @ t=1.2: unobserved
    tracker.update([], 3, 1.2)

    # Delayed detection arrived for Frame 2 @ t=1.1 (e.g. cup was actually detected at 1.01, 0, 0)
    obs_delayed = _create_obs("c2", 2, 1.1, [1.01, 0.0, 0.0], class_name="cup")
    rolled_tracks = tracker.update_delayed(
        observations=[obs_delayed],
        obs_frame_index=2,
        obs_timestamp=1.1,
        current_frame_index=3,
        current_timestamp=1.2,
    )

    assert len(rolled_tracks) == 1
    # Observation count should be 2 because delayed frame 2 observation was retroactively applied
    assert rolled_tracks[0].observation_count == 2
    assert rolled_tracks[0].class_name == "cup"
