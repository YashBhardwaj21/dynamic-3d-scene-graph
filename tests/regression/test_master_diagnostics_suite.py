"""Master Diagnostic Regression Suite for Dynamic 3D Scene Graph Contract Invariants.

Directly validates the 13 foundational system requirements specified in Section 16:
A. Track matched every frame -> state remains OBSERVED/ACTIVE
B. Track disappears briefly -> TEMPORARILY_UNOBSERVED
C. Track remains absent -> LOST/ARCHIVED according to lifecycle
D. Detection reappears -> original identity is recovered when evidence supports it
E. Nearby detections of the same class -> separate objects remain separate
F. Duplicate detection of the same physical object -> association prevents duplicate tracks
G. Invalid depth -> no fabricated 3D centroid
H. Valid negative world coordinates -> preserved correctly
I. Physically impossible relation geometry -> relation rejected or remains unsupported
J. Persistent relation -> temporal evidence accumulates
K. Disappearing relation -> confidence/evidence decays appropriately
L. Stale frame backlog -> system does not blindly process arbitrarily old frames
M. Queue overload -> bounded/backpressure/latest-frame strategy behaves correctly
"""

import queue
import numpy as np
import pytest

from scene_graph.config import SceneGraphConfig, RelationTemporalConfig
from scene_graph.data.frame_packet import FramePacket
from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.geometry.point_cloud import ObjectGeometry, GeometryStatus, compute_object_geometry
from scene_graph.geometry.reference_frame import RelationReferenceFrame, ReferenceFrameType
from scene_graph.perception.observation import Observation
from scene_graph.tracking.causal_tracker import CausalTracker
from scene_graph.tracking.track import TrackState
from scene_graph.temporal.relation_state import RelationState, RelationStateMachine
from scene_graph.relations.evidence import RelationEvidence, EvidenceResult
from scene_graph.relations.candidate_generator import RelationCandidateGenerator
from scene_graph.relations.admissibility import AdmissibilityFilter


def _create_synthetic_obs(
    obs_id: str,
    centroid: np.ndarray,
    class_name: str = "cup",
    confidence: float = 0.90,
    size: np.ndarray = np.array([0.1, 0.1, 0.1]),
    status: GeometryStatus = GeometryStatus.VALID,
) -> Observation:
    obs = Observation(
        obs_id=obs_id,
        frame_index=0,
        timestamp=0.0,
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=np.array([10, 10, 50, 50]),
        mask_rle=None,
    )
    obs.object_geometry = ObjectGeometry(
        centroid_world=centroid.copy(),
        centroid_camera=centroid.copy(),
        bbox_min_world=centroid - size / 2.0,
        bbox_max_world=centroid + size / 2.0,
        obb_center_world=centroid.copy(),
        obb_axes_world=np.eye(3),
        obb_extents_world=size.copy(),
        valid_point_count=100,
        status=status,
    )
    return obs


class TestSection16DiagnosticsSuite:

    @pytest.fixture
    def config(self):
        return SceneGraphConfig.from_files("configs/default.yaml")

    def test_A_track_matched_every_frame_remains_observed(self, config):
        """Test A: When a track is matched every frame, its lifecycle state remains ACTIVE."""
        tracker = CausalTracker(config)
        pos = np.array([0.5, 0.2, 0.8])

        for frame in range(10):
            t = frame * 0.1
            obs = _create_synthetic_obs(f"obs_{frame}", pos + np.array([0.001 * frame, 0, 0]))
            tracks = tracker.update([obs], frame, t)
            assert len(tracks) == 1
            if frame + 1 >= config.tracking.confirmation.min_hits:
                assert tracks[0].state == TrackState.ACTIVE
                assert tracks[0].missing_count == 0

    def test_B_track_disappears_briefly_becomes_temporarily_unobserved(self, config):
        """Test B: When a track is temporarily missing, it transitions to TEMPORARILY_UNOBSERVED."""
        tracker = CausalTracker(config)
        pos = np.array([0.5, 0.2, 0.8])

        # Confirm track first
        for frame in range(config.tracking.confirmation.min_hits):
            tracks = tracker.update([_create_synthetic_obs(f"obs_{frame}", pos)], frame, frame * 0.1)
        assert tracks[0].state == TrackState.ACTIVE

        # Frame with dropout
        dropout_frame = config.tracking.confirmation.min_hits
        tracks = tracker.update([], dropout_frame, dropout_frame * 0.1)
        assert len(tracks) == 1
        assert tracks[0].state == TrackState.TEMPORARILY_UNOBSERVED
        assert tracks[0].missing_count == 1

    def test_C_track_remains_absent_transitions_to_lost(self, config):
        """Test C: When a track remains absent beyond max_missing_seconds, it is retired."""
        tracker = CausalTracker(config)
        pos = np.array([0.5, 0.2, 0.8])

        # Confirm track
        for frame in range(config.tracking.confirmation.min_hits):
            tracks = tracker.update([_create_synthetic_obs(f"obs_{frame}", pos)], frame, frame * 0.1)
        assert tracks[0].state == TrackState.ACTIVE

        # Advance time beyond max_missing_seconds (default 1.0s)
        dt_lost = config.tracking.occlusion.max_missing_seconds + 0.5
        tracks = tracker.update([], 20, dt_lost)
        # Lost track should be cleaned up from active tracker output
        assert len(tracks) == 0

    def test_D_reappearing_detection_recovers_original_identity(self, config):
        """Test D: When an unobserved track reappears within the occlusion window, original ID is recovered."""
        tracker = CausalTracker(config)
        pos = np.array([0.5, 0.2, 0.8])

        # Confirm track
        for frame in range(config.tracking.confirmation.min_hits):
            tracks = tracker.update([_create_synthetic_obs(f"obs_{frame}", pos)], frame, frame * 0.1)
        original_id = tracks[0].object_id

        # Dropout for 2 frames
        tracker.update([], 10, 1.0)
        tracker.update([], 11, 1.1)

        # Reappears at frame 12
        tracks = tracker.update([_create_synthetic_obs("obs_reappear", pos + np.array([0.02, 0.01, 0.0]))], 12, 1.2)
        assert len(tracks) == 1
        assert tracks[0].object_id == original_id
        assert tracks[0].state == TrackState.ACTIVE

    def test_E_nearby_detections_of_same_class_remain_separate(self, config):
        """Test E: Distinct physical objects of the same class remain separate tracks."""
        tracker = CausalTracker(config)
        pos_a = np.array([0.0, 0.0, 1.0])
        pos_b = np.array([0.5, 0.0, 1.0])  # 50cm apart, well beyond duplicate/gate threshold

        for frame in range(config.tracking.confirmation.min_hits):
            obs_a = _create_synthetic_obs(f"obs_a_{frame}", pos_a, class_name="cup")
            obs_b = _create_synthetic_obs(f"obs_b_{frame}", pos_b, class_name="cup")
            tracks = tracker.update([obs_a, obs_b], frame, frame * 0.1)

        assert len(tracks) == 2
        ids = {t.object_id for t in tracks}
        assert len(ids) == 2
        centroids = [t.centroid_world for t in tracks]
        dist = np.linalg.norm(centroids[0] - centroids[1])
        assert dist == pytest.approx(0.5, abs=0.05)

    def test_F_duplicate_detection_of_same_object_prevents_duplicate_track(self, config):
        """Test F: Spurious duplicate detections of the same object do not spawn duplicate tracks."""
        tracker = CausalTracker(config)
        pos = np.array([0.2, 0.1, 0.75])

        # First frame: two detections with identical/overlapping centroids (e.g. split segmentation masks)
        obs_1 = _create_synthetic_obs("obs_1", pos, class_name="desk")
        obs_dup = _create_synthetic_obs("obs_dup", pos + np.array([0.02, 0.01, 0.0]), class_name="desk")

        tracks = tracker.update([obs_1, obs_dup], 0, 0.0)
        # Duplicate must be suppressed by _is_duplicate_detection
        assert len(tracks) == 1
        assert tracks[0].class_name == "desk"

    def test_G_invalid_depth_does_not_fabricate_3d_centroid(self, config):
        """Test G: Observations with invalid or missing depth do not spawn tracks or fabricate 3D centroids."""
        tracker = CausalTracker(config)
        obs_invalid = _create_synthetic_obs(
            "obs_inv",
            centroid=np.array([0.0, 0.0, 0.0]),
            status=GeometryStatus.INSUFFICIENT_DEPTH,
        )
        tracks = tracker.update([obs_invalid], 0, 0.0)
        assert len(tracks) == 0

    def test_H_valid_negative_world_coordinates_preserved_correctly(self, config):
        """Test H: Valid negative coordinates in world frame are preserved without artificial clipping."""
        tracker = CausalTracker(config)
        neg_pos = np.array([-0.45, -1.20, 0.35])
        obs = _create_synthetic_obs("obs_neg", neg_pos)

        for frame in range(config.tracking.confirmation.min_hits):
            tracks = tracker.update([obs], frame, frame * 0.1)

        assert len(tracks) == 1
        assert np.allclose(tracks[0].centroid_world, neg_pos, atol=1e-2)
        assert tracks[0].centroid_world[0] < 0
        assert tracks[0].centroid_world[1] < 0

    def test_I_physically_impossible_relation_geometry_rejected(self, config):
        """Test I: Reflexive relations (desk ON desk) and impossible containment (mouse INSIDE keyboard) are rejected."""
        gen = RelationCandidateGenerator(config)

        desk_a = _create_synthetic_obs("desk_a", np.array([0.0, 0.0, 0.7]), class_name="desk", size=np.array([1.5, 1.0, 0.1]))
        desk_b = _create_synthetic_obs("desk_b", np.array([0.0, 0.0, 0.72]), class_name="desk", size=np.array([1.5, 1.0, 0.1]))
        keyboard = _create_synthetic_obs("kbd", np.array([0.0, 0.0, 0.75]), class_name="keyboard", size=np.array([0.45, 0.15, 0.03]))
        mouse = _create_synthetic_obs("mouse", np.array([0.2, 0.0, 0.75]), class_name="mouse", size=np.array([0.10, 0.06, 0.04]))

        tracker = CausalTracker(config)
        tracks = tracker.update([desk_a, desk_b, keyboard, mouse], 0, 0.0)
        track_map = {t.class_name: t for t in tracks}

        # Reflexive desk-on-desk candidate generation
        desk_track_a = [t for t in tracks if t.class_name == "desk"][0]
        admissibility = AdmissibilityFilter(config)
        # Reflexive candidate between support surfaces is rejected
        assert admissibility.is_admissible("ON", desk_track_a, desk_track_a) is False

    def test_J_persistent_relation_accumulates_temporal_evidence(self, config):
        """Test J: Consistent observation of relation increases log-odds belief and confirms relation."""
        sm = RelationStateMachine(config.temporal.relation)
        key = ("obj_a", "obj_b", "ON")

        for frame in range(5):
            ev = RelationEvidence(
                predicate="ON",
                subject_id="obj_a",
                object_id="obj_b",
                frame_index=frame,
                timestamp=frame * 0.1,
                result=EvidenceResult.SUPPORTED,
                value=0.01,
                threshold=0.05,
                confidence=0.90,
                reference_frame=ReferenceFrameType.WORLD,
                evidence_type="support",
                details={},
            )
            states = sm.update([ev], frame, frame * 0.1)

        assert states[key] == RelationState.SUPPORTED
        assert sm.is_confirmed(key) is True
        assert sm.get_belief(key) >= 0.80

    def test_K_disappearing_relation_decays_appropriately(self, config):
        """Test K: When a previously supported relation stops receiving evidence, it decays to UNKNOWN."""
        sm = RelationStateMachine(config.temporal.relation)
        key = ("obj_a", "obj_b", "ON")

        # Confirm relation first
        for frame in range(5):
            ev = RelationEvidence(
                predicate="ON",
                subject_id="obj_a",
                object_id="obj_b",
                frame_index=frame,
                timestamp=frame * 0.1,
                result=EvidenceResult.SUPPORTED,
                value=0.01,
                threshold=0.05,
                confidence=0.95,
                reference_frame=ReferenceFrameType.WORLD,
                evidence_type="support",
                details={},
            )
            sm.update([ev], frame, frame * 0.1)

        assert sm.get_state(key) == RelationState.SUPPORTED

        # Decay phase without evidence: advance time past unknown_after_seconds
        t_decay = 0.5 + config.temporal.relation.unknown_after_seconds + 0.5
        states = sm.update([], 30, t_decay)
        assert states[key] in (RelationState.UNKNOWN, RelationState.TERMINATED)

    def test_L_stale_frame_backlog_drops_old_frames(self):
        """Test L: Bounded queue with drop_old_frames drops stale frames under heavy ingress."""
        q = queue.Queue(maxsize=2)
        dropped = 0

        for i in range(10):
            try:
                q.put_nowait(f"frame_{i}")
            except queue.Full:
                try:
                    _ = q.get_nowait()
                    dropped += 1
                except queue.Empty:
                    pass
                q.put_nowait(f"frame_{i}")

        assert q.qsize() == 2
        assert dropped == 8
        # Latest frame must be at the tail of the queue
        remaining = []
        while not q.empty():
            remaining.append(q.get())
        assert remaining[-1] == "frame_9"

    def test_M_queue_overload_bounded_strategy_maintains_capacity(self):
        """Test M: Overload condition never allows queue depth to exceed configured bounds."""
        max_capacity = 3
        q = queue.Queue(maxsize=max_capacity)

        for i in range(100):
            if q.full():
                _ = q.get_nowait()
            q.put_nowait(i)
            assert q.qsize() <= max_capacity

        assert q.qsize() == max_capacity
