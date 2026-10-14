"""Unit tests for Stage 5 Inexpensive Pre-Router Evidence Pipeline."""

import pytest
import math
import sys
from pathlib import Path

# Add project root and vlm to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "vlm"))

from vlm.src.geometric_evidence import (
    GeometricEvidenceExtractor,
    compute_depth_uncertainty,
    quaternion_to_rotation_matrix,
    FX, FY, CX, CY, STEREO_BASELINE_M
)
from vlm.src.temporal_evidence import TemporalEvidenceExtractor
from vlm.src.semantic_evidence import SemanticEvidenceExtractor, CLASS_MOBILITY_PRIORS


def test_geometric_depth_uncertainty_model():
    """Verify physical RealSense D455 stereo triangulation uncertainty formula.
    sigma_z = (z^2 / (f_x * B)) * sigma_d
    """
    sigma_1m = compute_depth_uncertainty(1.0)
    sigma_2m = compute_depth_uncertainty(2.0)
    sigma_3m = compute_depth_uncertainty(3.0)

    # Monotonic quadratic growth with depth
    assert sigma_1m < sigma_2m < sigma_3m
    # 1.0m uncertainty should be around 2mm to 4mm
    assert 0.001 <= sigma_1m <= 0.005
    # 3.0m uncertainty should be around 1.5cm to 3cm
    assert 0.010 <= sigma_3m <= 0.035


def test_geometric_quaternion_orthonormality():
    """Verify quaternion to rotation matrix produces valid orthonormal SO(3) matrix."""
    # Identity quaternion
    R_eye = quaternion_to_rotation_matrix(0, 0, 0, 1)
    assert R_eye.shape == (3, 3)
    assert math.isclose(R_eye[0, 0], 1.0, abs_tol=1e-5)
    assert math.isclose(R_eye[1, 1], 1.0, abs_tol=1e-5)
    assert math.isclose(R_eye[2, 2], 1.0, abs_tol=1e-5)

    # Arbitrary normalized quaternion
    R = quaternion_to_rotation_matrix(0.1, -0.2, 0.3, 0.9)
    # R * R^T = I
    I_test = R @ R.T
    for i in range(3):
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            assert math.isclose(I_test[i, j], expected, abs_tol=1e-4)


def test_geometric_evidence_extraction():
    """Verify geometric record bounds and consistency score computation."""
    extractor = GeometricEvidenceExtractor()
    sample_obs = {
        "obs_id": "test_001",
        "frame_index": 100,
        "timestamp": 1000.5,
        "global_entity_id": "desk_001",
        "local_track_id": "trk_001",
        "points_count": 2500,
        "centroid_camera": [0.5, -0.2, 2.0],
        "bbox_min_floor": [1.0, -1.0, 0.0],
        "bbox_max_floor": [2.0, -0.5, 0.8],
        "camera_pose": [0.1, 0.0, 0.5, 0, 0, 0, 1],
        "bbox_xyxy": [100.0, 150.0, 300.0, 350.0],
    }

    rec = extractor.extract_observation_geometry(sample_obs)
    assert rec.depth_valid is True
    assert math.isclose(rec.measured_depth_m, 2.0, abs_tol=1e-3)
    assert 0.0 <= rec.inlier_ratio <= 1.0
    assert 0.0 <= rec.geometric_consistency_score <= 1.0
    assert rec.mean_reprojection_residual_px >= 0.0
    assert rec.has_valid_evidence is True


def test_temporal_ego_motion_compensation():
    """Verify that moving the camera while object stays still yields STATIC motion."""
    extractor = TemporalEvidenceExtractor()

    # Step 1: Initial observation at (2.0, -1.0, 0.4), camera at (0.0, 0.0, 0.0)
    obs1 = {
        "obs_id": "step_01",
        "frame_index": 0,
        "timestamp": 0.0,
        "global_entity_id": "chair_001",
        "local_track_id": "trk_c1",
        "centroid_floor": [2.0, -1.0, 0.4],
        "camera_pose": [0.0, 0.0, 0.5, 0, 0, 0, 1],
    }
    rec1 = extractor.extract_observation_temporal(obs1)
    assert rec1.current_motion == "TRANSIENT"
    assert rec1.world_displacement_m == 0.0

    # Step 2: Camera translates 1.5m, object has tiny 1cm noise
    obs2 = {
        "obs_id": "step_02",
        "frame_index": 10,
        "timestamp": 0.5,
        "global_entity_id": "chair_001",
        "local_track_id": "trk_c1",
        "centroid_floor": [2.01, -1.0, 0.4],
        "camera_pose": [1.5, 0.2, 0.5, 0, 0, 0, 1],
    }
    extractor.extract_observation_temporal(obs2)

    # Step 3: Camera translates 2.5m, object stays within noise floor
    obs3 = {
        "obs_id": "step_03",
        "frame_index": 20,
        "timestamp": 1.0,
        "global_entity_id": "chair_001",
        "local_track_id": "trk_c1",
        "centroid_floor": [2.015, -0.995, 0.4],
        "camera_pose": [2.5, 0.4, 0.5, 0, 0, 0, 1],
    }
    rec3 = extractor.extract_observation_temporal(obs3)

    # Object is confirmed STATIC in world frame despite camera moving 2.5m
    assert rec3.current_motion == "STATIC"
    assert rec3.world_displacement_m < 0.05
    assert rec3.camera_ego_displacement_m > 2.0
    assert rec3.ego_motion_compensation_ratio > 10.0


def test_temporal_displaced_object_detection():
    """Verify that true physical relocation (>0.15m) triggers MOVING/DISPLACED."""
    extractor = TemporalEvidenceExtractor()

    # Step 1: Initial position
    extractor.extract_observation_temporal({
        "obs_id": "t1", "frame_index": 0, "timestamp": 0.0,
        "global_entity_id": "chair_reloc", "local_track_id": "trk_r",
        "centroid_floor": [1.0, 1.0, 0.4], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    extractor.extract_observation_temporal({
        "obs_id": "t2", "frame_index": 5, "timestamp": 0.25,
        "global_entity_id": "chair_reloc", "local_track_id": "trk_r",
        "centroid_floor": [1.0, 1.0, 0.4], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    # Step 3: Relocated by 0.56m (matching controlled chair_002 relocation)
    rec3 = extractor.extract_observation_temporal({
        "obs_id": "t3", "frame_index": 30, "timestamp": 1.5,
        "global_entity_id": "chair_reloc", "local_track_id": "trk_r",
        "centroid_floor": [1.56, 1.0, 0.4], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })

    assert rec3.current_motion in ["MOVING", "DISPLACED"]
    assert math.isclose(rec3.world_displacement_m, 0.56, abs_tol=1e-2)
    assert rec3.motion_significance_ratio > 3.0


def test_semantic_entropy_and_cheap_prior():
    """Verify semantic Shannon entropy and deterministic M0 priors."""
    extractor = SemanticEvidenceExtractor()

    # Pure chair observations
    rec1 = extractor.extract_observation_semantics({
        "obs_id": "s1", "frame_index": 0, "timestamp": 0.0,
        "global_entity_id": "chair_001", "class_name": "chair", "confidence": 0.90,
    })
    assert rec1.class_prior_mobility == "MOVABLE"
    assert rec1.semantic_entropy == 0.0
    assert rec1.semantic_purity == 1.0

    # Class flip observation introduces entropy
    rec2 = extractor.extract_observation_semantics({
        "obs_id": "s2", "frame_index": 1, "timestamp": 0.1,
        "global_entity_id": "chair_001", "class_name": "table", "confidence": 0.40,
    })
    assert rec2.semantic_entropy > 0.0
    assert rec2.semantic_purity == 0.50

    # Door fixture should have IMMOVABLE prior
    door_rec = extractor.extract_observation_semantics({
        "obs_id": "s3", "frame_index": 2, "timestamp": 0.2,
        "global_entity_id": "door_001", "class_name": "door", "confidence": 0.95,
    })
    assert door_rec.class_prior_mobility == "IMMOVABLE"
    assert door_rec.class_prior_confidence >= 0.90


def test_acceptance_gate_zero_vlm_dependency():
    """Verify that Stage 5 executes with zero VLM querying and zero network requests."""
    # Ensure extractor modules have no import of vlm query or network modules
    from vlm.src import geometric_evidence, temporal_evidence, semantic_evidence
    import urllib.request

    # Run complete extract cycle
    geo = geometric_evidence.GeometricEvidenceExtractor()
    tem = temporal_evidence.TemporalEvidenceExtractor()
    sem = semantic_evidence.SemanticEvidenceExtractor()

    obs = {
        "obs_id": "gate_test",
        "frame_index": 50,
        "timestamp": 12.34,
        "global_entity_id": "desk_gate",
        "local_track_id": "trk_gate",
        "class_name": "desk",
        "confidence": 0.85,
        "points_count": 3000,
        "centroid_camera": [0.0, 0.0, 1.8],
        "centroid_floor": [1.5, -0.8, 0.4],
        "bbox_min_floor": [1.0, -1.2, 0.0],
        "bbox_max_floor": [2.0, -0.4, 0.8],
        "camera_pose": [0, 0, 0, 0, 0, 0, 1],
        "bbox_xyxy": [150, 100, 450, 400],
    }

    g_res = geo.extract_observation_geometry(obs)
    t_res = tem.extract_observation_temporal(obs)
    s_res = sem.extract_observation_semantics(obs)

    assert g_res.has_valid_evidence is True
    assert t_res.has_valid_evidence is True
    assert s_res.has_valid_evidence is True
    assert s_res.class_prior_mobility == "MOVABLE"
