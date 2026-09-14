import pytest
import numpy as np

from scene_graph.config import SceneGraphConfig
from scene_graph.perception.observation import Observation
from scene_graph.geometry.point_cloud import ObjectGeometry
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


@pytest.fixture
def wide_gate_config():
    """Config with a wide association gate to allow more matches in stress tests."""
    return SceneGraphConfig.model_validate({
        "tracking": {
            "confirmation": {"min_hits": 1},
            "occlusion": {"max_missing_seconds": 1.0},
            "association": {"max_distance_m": 2.0},
        }
    })


def _make_obs(obs_id, frame_index, timestamp, centroid, class_name="monitor"):
    """Helper to create an Observation with attached ObjectGeometry."""
    centroid = np.asarray(centroid, dtype=np.float64)
    obs = Observation(
        obs_id=obs_id,
        frame_index=frame_index,
        timestamp=timestamp,
        class_name=class_name,
        confidence=0.9,
        bbox_xyxy=np.zeros(4),
        mask_rle=None,
    )
    obs.object_geometry = ObjectGeometry(
        centroid_world=centroid.copy(),
        centroid_camera=centroid.copy(),
        bbox_min_world=centroid - 0.01,
        bbox_max_world=centroid + 0.01,
        valid_point_count=100,
    )
    return obs


def test_causal_tracker_lifecycle(tracker_config):
    tracker = CausalTracker(tracker_config)

    # Frame 1: New observation
    obs1 = _make_obs("obs_1", 1, 1.0, [1.0, 2.0, 3.0])
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
    obs2 = _make_obs("obs_2", 2, 1.1, [1.01, 2.0, 3.0])
    tracks = tracker.update([obs2], 2, 1.1)
    assert len(tracks) == 1
    assert tracks[0].state == TrackState.CANDIDATE
    assert tracks[0].observation_count == 2

    # Frame 3: Same observation (confirms after min_hits=3)
    obs3 = _make_obs("obs_3", 3, 1.2, [1.02, 2.0, 3.0])
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


class TestInfeasibleCostMatrix:
    """Regression tests for the SciPy linear_sum_assignment infeasibility bug.

    The old code passed the feasible sub-matrix directly to linear_sum_assignment.
    When class-gating or distance-gating removed only *some* cells in a row/column
    that was otherwise feasible (i.e. the row had at least one finite cell but also
    inf cells), the solver raised ValueError: cost matrix is infeasible.
    """

    def test_mixed_class_no_crash(self, tracker_config):
        """Two existing tracks of different classes, two new observations of mixed classes.

        This creates a feasible sub-matrix with inf entries (class mismatch)
        that previously caused the solver to crash.
        """
        tracker = CausalTracker(tracker_config)

        # Seed: frame 1 — one monitor, one keyboard
        obs_monitor = _make_obs("m1", 1, 1.0, [1.0, 0.0, 0.0], class_name="monitor")
        obs_keyboard = _make_obs("k1", 1, 1.0, [2.0, 0.0, 0.0], class_name="keyboard")
        tracks = tracker.update([obs_monitor, obs_keyboard], 1, 1.0)
        assert len(tracks) == 2

        # Frame 2 — same two objects, slightly moved
        obs_monitor2 = _make_obs("m2", 2, 1.1, [1.01, 0.0, 0.0], class_name="monitor")
        obs_keyboard2 = _make_obs("k2", 2, 1.1, [2.01, 0.0, 0.0], class_name="keyboard")

        # This MUST NOT raise ValueError: cost matrix is infeasible
        tracks = tracker.update([obs_monitor2, obs_keyboard2], 2, 1.1)
        assert len(tracks) == 2

        # Verify correct class-to-track matching
        track_classes = {t.object_id: t.class_name for t in tracks}
        assert "monitor" in track_classes.values()
        assert "keyboard" in track_classes.values()

    def test_all_inf_returns_no_matches(self, tracker_config):
        """When every entry in the cost matrix is inf, return all unmatched."""
        tracker = CausalTracker(tracker_config)

        # Seed a "monitor" track
        obs1 = _make_obs("m1", 1, 1.0, [0.0, 0.0, 0.0], class_name="monitor")
        tracker.update([obs1], 1, 1.0)

        # Frame 2: observation of a completely different class (no possible match)
        obs2 = _make_obs("k1", 2, 1.1, [0.0, 0.0, 0.0], class_name="keyboard")
        tracks = tracker.update([obs2], 2, 1.1)

        # Should create a new track for keyboard, monitor should be missing
        keyboard_tracks = [t for t in tracks if t.class_name == "keyboard"]
        monitor_tracks = [t for t in tracks if t.class_name == "monitor"]
        assert len(keyboard_tracks) == 1
        assert len(monitor_tracks) == 1
        assert monitor_tracks[0].missing_count == 1

    def test_single_obs_multiple_tracks_different_classes(self, tracker_config):
        """One observation with multiple tracks of different classes.

        Only the same-class track should match.
        """
        tracker = CausalTracker(tracker_config)

        # Seed: three tracks of different classes at the same location
        obs_a = _make_obs("a1", 1, 1.0, [1.0, 0.0, 0.0], class_name="cup")
        obs_b = _make_obs("b1", 1, 1.0, [1.05, 0.0, 0.0], class_name="bottle")
        obs_c = _make_obs("c1", 1, 1.0, [0.95, 0.0, 0.0], class_name="keyboard")
        tracker.update([obs_a, obs_b, obs_c], 1, 1.0)

        # Frame 2: only a cup observation near the original location
        obs_cup2 = _make_obs("a2", 2, 1.1, [1.01, 0.0, 0.0], class_name="cup")
        tracks = tracker.update([obs_cup2], 2, 1.1)

        # Cup track matched, others missing
        cup_tracks = [t for t in tracks if t.class_name == "cup"]
        assert len(cup_tracks) == 1
        assert cup_tracks[0].observation_count == 2

        others = [t for t in tracks if t.class_name != "cup"]
        for t in others:
            assert t.missing_count >= 1

    def test_stress_multiframe_mixed_classes(self, wide_gate_config):
        """Stress test: 50 frames with objects appearing/disappearing across 4 classes.

        Ensures zero exceptions through many association rounds with partial gating.
        """
        tracker = CausalTracker(wide_gate_config)
        rng = np.random.RandomState(42)
        classes = ["monitor", "keyboard", "cup", "bottle"]

        for frame in range(50):
            timestamp = float(frame) * 0.1
            n_obs = rng.randint(1, 8)
            observations = []
            for i in range(n_obs):
                cls = classes[rng.randint(0, len(classes))]
                centroid = rng.randn(3) * 0.5
                obs = _make_obs(f"f{frame}_o{i}", frame, timestamp, centroid, class_name=cls)
                observations.append(obs)

            # This must never raise
            tracks = tracker.update(observations, frame, timestamp)
            assert isinstance(tracks, list)

    def test_partial_gating_correct_assignment(self, tracker_config):
        """Verify correct assignment when one observation can match one of two
        same-class tracks but not the other (distance gating).
        """
        tracker = CausalTracker(tracker_config)

        # Seed: two monitors far apart
        obs_a = _make_obs("a", 1, 1.0, [0.0, 0.0, 0.0], class_name="monitor")
        obs_b = _make_obs("b", 1, 1.0, [10.0, 0.0, 0.0], class_name="monitor")
        tracker.update([obs_a, obs_b], 1, 1.0)

        # Frame 2: one monitor near track A only (within 0.25m gate)
        obs_near_a = _make_obs("c", 2, 1.1, [0.05, 0.0, 0.0], class_name="monitor")
        tracks = tracker.update([obs_near_a], 2, 1.1)

        # Track near A should match, track B should be missing
        matched_track = [t for t in tracks if t.observation_count == 2]
        missed_track = [t for t in tracks if t.missing_count >= 1]
        assert len(matched_track) == 1
        assert len(missed_track) == 1
        # The matched one should be close to origin
        assert np.linalg.norm(matched_track[0].centroid_world) < 0.5

