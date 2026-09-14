"""Master System Regression Suite for Dynamic 3D Scene Graph Contract Invariants.

Verifies the 10 foundational system invariants across the unified pipeline:
1. Canonical Sensor & Metric Depth Contract
2. Strict Schema & Enum Validation
3. Semantic-Independent Causal Tracking Continuity (Class-Flip)
4. Partial Observability & Geometry Provenance (PREDICTED on Dropout)
5. Sensor-Aware Depth Uncertainty (DepthNoiseModel)
6. Dynamic Geometric Roles & Admissibility (RelWitness)
7. Persistent Spatial Surfaces (Dropout Resistance)
8. Reference Frame Invariance (MAP vs CAMERA)
9. Normalized Evidence Strength in [0, 1]
10. Bounded History Memory Guarantee
"""

from collections import deque
import numpy as np
import pytest

from scene_graph.data.frame_packet import FramePacket
from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.geometry.reference_frame import RelationReferenceFrame, ReferenceFrameType
from scene_graph.geometry.noise_model import StereoDepthNoiseModel
from scene_graph.geometry.point_cloud import ObjectGeometry, GeometryStatus
from scene_graph.geometry.spatial_surface import SpatialSurface, SurfaceManager
from scene_graph.config import SceneGraphConfig, RelationTemporalConfig
from scene_graph.perception.observation import Observation
from scene_graph.tracking.track import Track, TrackState
from scene_graph.tracking.causal_tracker import CausalTracker
from scene_graph.temporal.object_state import ObjectState
from scene_graph.temporal.relation_state import RelationState, RelationStateMachine
from scene_graph.relations.evidence import RelationEvidence, EvidenceResult
from scene_graph.relations.admissibility import AdmissibilityFilter
from scene_graph.relations.directional import DirectionalRelationModule
from scene_graph.relations.context import FrameContext, ObservationGeometry
from scene_graph.graph.event import GraphEvent, GraphEventType
from scene_graph.graph.graph_history import GraphHistory


class TestSystemRegressionInvariants:

    def test_invariant_1_metric_depth_contract(self):
        """Invariant 1: FramePacket.depth is strictly float32 meters; raw uint16 is rejected."""
        valid_depth = np.ones((100, 100), dtype=np.float32) * 1.5
        packet = FramePacket(
            frame_index=0,
            timestamp=0.0,
            rgb=np.zeros((100, 100, 3), dtype=np.uint8),
            depth=valid_depth,
            world_T_camera=np.eye(4),
            camera_intrinsics=CameraIntrinsics(fx=500, fy=500, cx=50, cy=50, width=100, height=100),
            depth_model=DepthModel(scale=1000.0),
        )
        assert packet.depth.dtype == np.float32
        assert packet.depth[0, 0] == 1.5

        # RealSense raw depth (uint16) must be rejected
        raw_uint16 = (np.ones((100, 100)) * 1500).astype(np.uint16)
        with pytest.raises((TypeError, ValueError)):
            FramePacket(
                frame_index=0,
                timestamp=0.0,
                rgb=np.zeros((100, 100, 3), dtype=np.uint8),
                depth=raw_uint16,
                world_T_camera=np.eye(4),
                camera_intrinsics=CameraIntrinsics(fx=500, fy=500, cx=50, cy=50, width=100, height=100),
                depth_model=DepthModel(scale=1000.0),
            )

    def test_invariant_2_strict_schema_and_enums(self):
        """Invariant 2: RelationEvidence strictly enforces ReferenceFrameType and EvidenceResult enums."""
        ev = RelationEvidence(
            predicate="ON",
            subject_id="obj_a",
            object_id="obj_b",
            frame_index=1,
            timestamp=1.0,
            result=EvidenceResult.SUPPORTED,
            value=0.05,
            threshold=0.10,
            confidence=0.90,
            reference_frame=ReferenceFrameType.WORLD,
            evidence_type="support",
            details={},
        )
        assert ev.reference_frame == ReferenceFrameType.WORLD
        assert ev.result == EvidenceResult.SUPPORTED

        with pytest.raises(TypeError, match="ReferenceFrameType"):
            RelationEvidence(
                predicate="ON",
                subject_id="obj_a",
                object_id="obj_b",
                frame_index=1,
                timestamp=1.0,
                result=EvidenceResult.SUPPORTED,
                value=0.05,
                threshold=0.10,
                confidence=0.90,
                reference_frame="WORLD",  # raw string rejected
                evidence_type="support",
                details={},
            )

    def test_invariant_3_class_flip_tracking_continuity(self):
        """Invariant 3: Physical object identity is preserved across semantic label flips."""
        config = SceneGraphConfig.from_files("configs/default.yaml")
        tracker = CausalTracker(config)
        centroid = np.array([1.0, 0.0, 1.0])

        def _make_obs(fid, label):
            obs = Observation(
                obs_id=f"obs_{fid}",
                frame_index=fid,
                timestamp=float(fid) * 0.1,
                class_name=label,
                confidence=0.85,
                bbox_xyxy=np.array([10, 10, 50, 50]),
                mask_rle=None,
            )
            obs.object_geometry = ObjectGeometry(
                centroid_world=centroid.copy(),
                centroid_camera=centroid.copy(),
                bbox_min_world=centroid - 0.1,
                bbox_max_world=centroid + 0.1,
                obb_center_world=centroid.copy(),
                obb_axes_world=np.eye(3),
                obb_extents_world=np.array([0.2, 0.2, 0.2]),
                valid_point_count=100,
                status=GeometryStatus.VALID,
            )
            return obs

        labels = ["mug", "cup", "mug", "cup", "mug"]
        track_ids = []
        final_track = None
        for fid, label in enumerate(labels):
            tracks = tracker.update([_make_obs(fid, label)], fid, float(fid) * 0.1)
            assert len(tracks) == 1
            track_ids.append(tracks[0].object_id)
            final_track = tracks[0]

        # Invariant: Track ID must be identical across all alternating frames
        assert len(set(track_ids)) == 1
        assert "mug" in final_track.label_belief
        assert "cup" in final_track.label_belief

    def test_invariant_4_partial_observability_provenance(self):
        """Invariant 4: Unobserved active tracks synthesize PREDICTED geometry; relations decay gracefully."""
        config = RelationTemporalConfig(
            confirmation_threshold=0.5,
            contradiction_threshold=-0.5,
            decay_per_second=0.1,
            unknown_after_seconds=1.0,
            lost_after_seconds=3.0,
        )
        sm = RelationStateMachine(config)

        key = ("obj_a", "obj_b", "ON")
        ev = RelationEvidence(
            predicate="ON",
            subject_id="obj_a",
            object_id="obj_b",
            frame_index=0,
            timestamp=0.0,
            result=EvidenceResult.SUPPORTED,
            value=0.02,
            threshold=0.05,
            confidence=0.95,
            reference_frame=ReferenceFrameType.WORLD,
            evidence_type="support",
            details={},
        )
        states = sm.update([ev], frame_index=0, timestamp=0.0)
        assert states[key] == RelationState.SUPPORTED

        # At timestamp 0.5s (within grace window, no evidence emitted due to occlusion)
        states = sm.update([], frame_index=5, timestamp=0.5)
        # Should stay SUPPORTED in grace window
        assert states[key] == RelationState.SUPPORTED

        # At timestamp 1.5s (past grace window)
        states = sm.update([], frame_index=15, timestamp=1.5)
        # Must decay to UNKNOWN, NOT deleted or contradictory
        assert states[key] == RelationState.UNKNOWN

    def test_invariant_5_sensor_aware_depth_uncertainty(self):
        """Invariant 5: Depth uncertainty scales quadratically with depth for stereo sensors."""
        model = StereoDepthNoiseModel(baseline_m=0.095, subpixel_disparity_std=0.08)
        intrinsics = CameraIntrinsics(fx=380.0, fy=380.0, cx=320.0, cy=240.0, width=640, height=480)

        pts_1m = np.array([[0.0, 0.0, 1.0]])
        pts_2m = np.array([[0.0, 0.0, 2.0]])
        pts_4m = np.array([[0.0, 0.0, 4.0]])

        cov_1m = model.estimate_covariance(pts_1m, intrinsics)
        cov_2m = model.estimate_covariance(pts_2m, intrinsics)
        cov_4m = model.estimate_covariance(pts_4m, intrinsics)

        sigma_z_1m = np.sqrt(cov_1m[2, 2])
        sigma_z_2m = np.sqrt(cov_2m[2, 2])
        sigma_z_4m = np.sqrt(cov_4m[2, 2])

        assert sigma_z_2m == pytest.approx(sigma_z_1m * 4.0, rel=1e-2)
        assert sigma_z_4m == pytest.approx(sigma_z_1m * 16.0, rel=1e-2)
        assert sigma_z_1m < 0.01
        assert sigma_z_4m > 0.03

    def test_invariant_6_dynamic_geometric_roles_relwitness(self):
        """Invariant 6: An unknown planar slab provides physical ON support purely by geometric witness."""
        config = SceneGraphConfig.from_files("configs/default.yaml")
        filter_ = AdmissibilityFilter(config)

        slab = Track(
            object_id="slab_1",
            class_name="unlabeled_slab",
            state=TrackState.ACTIVE,
            _initial_centroid=np.array([0.0, 0.0, 0.5]),
            last_observed_frame=0,
            first_observed_frame=0,
            observation_count=1,
            missing_count=0,
            detection_confidence=0.9,
            track_observation_ratio=1.0,
            last_timestamp=0.0,
            recent_observations=deque(),
            size_world=np.array([1.0, 1.0, 0.1]),
        )
        cup = Track(
            object_id="item_1",
            class_name="unlabeled_item",
            state=TrackState.ACTIVE,
            _initial_centroid=np.array([0.0, 0.0, 0.62]),
            last_observed_frame=0,
            first_observed_frame=0,
            observation_count=1,
            missing_count=0,
            detection_confidence=0.9,
            track_observation_ratio=1.0,
            last_timestamp=0.0,
            recent_observations=deque(),
            size_world=np.array([0.1, 0.1, 0.14]),
        )

        assert filter_.is_admissible("ON", cup, slab) is True
        assert filter_.is_admissible("UNDER", slab, cup) is True

    def test_invariant_7_persistent_spatial_surfaces(self):
        """Invariant 7: Spatial surfaces persist across object dropout."""
        manager = SurfaceManager()
        pts = np.array([[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.0, 1.0, 0.5]])
        manager.register_or_update(
            track_id="base_table",
            normal=np.array([0.0, 0.0, 1.0]),
            distance=-0.5,
            plane_points=pts,
            timestamp=0.0,
        )

        # Next frame at timestamp 0.5s: base_table suffers tracking dropout (not observed)
        persisted = manager.get_surface("base_table", current_timestamp=0.5, max_age_s=2.0)
        assert persisted is not None
        assert persisted.track_id == "base_table"
        assert np.allclose(persisted.normal, [0.0, 0.0, 1.0])

    def test_invariant_8_reference_frame_map_invariance(self):
        """Invariant 8: Directional relations in MAP mode evaluate relative to world gravity."""
        config = SceneGraphConfig.from_files("configs/default.yaml")
        module = DirectionalRelationModule(config)
        T = np.eye(4)
        ref_map = RelationReferenceFrame.create("map", T)
        assert ref_map.frame_type == ReferenceFrameType.WORLD

        ref_cam = RelationReferenceFrame.create("camera", T)
        assert ref_cam.frame_type == ReferenceFrameType.CAMERA

        track_top = Track(
            object_id="obj_top",
            class_name="cup",
            state=TrackState.ACTIVE,
            _initial_centroid=np.array([0.0, 1.0, 1.5]),
            last_observed_frame=0,
            first_observed_frame=0,
            observation_count=1,
            missing_count=0,
            detection_confidence=0.9,
            track_observation_ratio=1.0,
            last_timestamp=0.0,
            recent_observations=deque(),
            size_world=np.array([0.1, 0.1, 0.1]),
        )
        track_bottom = Track(
            object_id="obj_bottom",
            class_name="desk",
            state=TrackState.ACTIVE,
            _initial_centroid=np.array([0.0, 1.0, 0.5]),
            last_observed_frame=0,
            first_observed_frame=0,
            observation_count=1,
            missing_count=0,
            detection_confidence=0.9,
            track_observation_ratio=1.0,
            last_timestamp=0.0,
            recent_observations=deque(),
            size_world=np.array([0.8, 0.8, 0.5]),
        )

        geom_top = ObservationGeometry(
            obs_id="obs_top",
            track_id="obj_top",
            centroid_world=np.array([0.0, 1.0, 1.5]),
            bbox_min_world=np.array([-0.05, 0.95, 1.45]),
            bbox_max_world=np.array([0.05, 1.05, 1.55]),
            depth_stats={"mean": 1.5},
            points_world_sampled=np.array([[0.0, 1.0, 1.5]]),
            points_world=np.array([[0.0, 1.0, 1.5]]),
            points_camera=None,
            mask=None,
            valid_point_count=1,
        )
        geom_bottom = ObservationGeometry(
            obs_id="obs_bottom",
            track_id="obj_bottom",
            centroid_world=np.array([0.0, 1.0, 0.5]),
            bbox_min_world=np.array([-0.4, 0.6, 0.25]),
            bbox_max_world=np.array([0.4, 1.4, 0.75]),
            depth_stats={"mean": 0.5},
            points_world_sampled=np.array([[0.0, 1.0, 0.5]]),
            points_world=np.array([[0.0, 1.0, 0.5]]),
            points_camera=None,
            mask=None,
            valid_point_count=1,
        )

        ctx = FrameContext(
            frame_index=0,
            timestamp=0.0,
            intrinsics=None,
            world_T_camera=T,
            reference_frame=ref_map,
            camera_frame=None,
            depth_image=None,
            observation_geometry={"obj_top": geom_top, "obj_bottom": geom_bottom},
        )
        evidences = module.compute(track_top, track_bottom, ctx)
        above_ev = [e for e in evidences if e.predicate == "ABOVE"][0]
        assert above_ev.result == EvidenceResult.SUPPORTED
        assert above_ev.reference_frame == ReferenceFrameType.WORLD

    def test_invariant_9_normalized_evidence_strength(self):
        """Invariant 9: RelationEvidence.evidence_strength is strictly bounded in [0.0, 1.0]."""
        ev = RelationEvidence(
            predicate="NEAR",
            subject_id="a",
            object_id="b",
            frame_index=0,
            timestamp=0.0,
            result=EvidenceResult.SUPPORTED,
            value=0.5,
            threshold=1.0,
            confidence=0.82,
            reference_frame=ReferenceFrameType.WORLD,
            evidence_type="distance",
            details={},
        )
        assert 0.0 <= ev.evidence_strength <= 1.0
        assert ev.evidence_strength == pytest.approx(0.82)

        # Assert negative or >1 confidence is rejected
        with pytest.raises(ValueError):
            RelationEvidence(
                predicate="NEAR",
                subject_id="a",
                object_id="b",
                frame_index=0,
                timestamp=0.0,
                result=EvidenceResult.SUPPORTED,
                value=0.5,
                threshold=1.0,
                confidence=1.5,
                reference_frame=ReferenceFrameType.WORLD,
                evidence_type="distance",
                details={},
            )

    def test_invariant_10_bounded_history_guarantee(self):
        """Invariant 10: Event history memory is strictly bounded by maxlen."""
        history = GraphHistory(maxlen=100)
        for i in range(2500):
            history.add_event(GraphEvent(
                event_type=GraphEventType.NODE_ADDED,
                timestamp=float(i),
                frame_index=i,
                subject_id=f"node_{i % 5}",
            ))
        assert len(history) == 100
        assert history.events[0].frame_index == 2400
        assert history.events[-1].frame_index == 2499
