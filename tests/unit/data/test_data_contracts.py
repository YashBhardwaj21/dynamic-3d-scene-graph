import os
from pathlib import Path
import pytest
import numpy as np

from scene_graph.config import SceneGraphConfig
from scene_graph.relations.evidence import (
    RelationEvidence,
    EvidenceResult,
    ReferenceFrameType,
)
from scene_graph.geometry.point_cloud import compute_object_geometry, GeometryStatus
from scene_graph.geometry.camera import CameraIntrinsics


def _dummy_intrinsics():
    return CameraIntrinsics(fx=525.0, fy=525.0, cx=320.0, cy=240.0, width=640, height=480)


def test_relation_evidence_enum_enforcement():
    """Verify that RelationEvidence strictly rejects string values and enforces enums."""
    valid_kwargs = {
        "predicate": "ON",
        "subject_id": "obj_1",
        "object_id": "obj_2",
        "frame_index": 0,
        "timestamp": 1.0,
        "result": EvidenceResult.SUPPORTED,
        "value": 0.01,
        "threshold": 0.02,
        "confidence": 0.9,
        "reference_frame": ReferenceFrameType.WORLD,
        "evidence_type": "contact",
        "details": {},
    }

    # Valid instantiation
    ev = RelationEvidence(**valid_kwargs)
    assert ev.reference_frame == ReferenceFrameType.WORLD
    assert ev.result == EvidenceResult.SUPPORTED

    # Reject string reference_frame
    invalid_frame_kwargs = dict(valid_kwargs)
    invalid_frame_kwargs["reference_frame"] = "world"
    with pytest.raises(TypeError, match="reference_frame must be ReferenceFrameType enum"):
        RelationEvidence(**invalid_frame_kwargs)

    # Reject string result
    invalid_result_kwargs = dict(valid_kwargs)
    invalid_result_kwargs["result"] = "supported"
    with pytest.raises(TypeError, match="result must be EvidenceResult enum"):
        RelationEvidence(**invalid_result_kwargs)

    # Reject confidence out of bounds
    invalid_conf_kwargs = dict(valid_kwargs)
    invalid_conf_kwargs["confidence"] = 1.5
    with pytest.raises(ValueError, match="confidence must be.*in"):
        RelationEvidence(**invalid_conf_kwargs)


def test_measurement_noise_propagation_into_covariance():
    """Verify that measurement_noise_std_m propagates directly into position_covariance_world."""
    intrinsics = _dummy_intrinsics()
    H, W = 480, 640
    mask = np.zeros((H, W), dtype=bool)
    mask[200:250, 300:350] = True  # 50x50 object patch
    depth_m = np.ones((H, W), dtype=np.float32) * 2.0  # 2.0 meters flat plane
    pose = np.eye(4)

    # Run with noise std 0.01
    geom_low_noise = compute_object_geometry(
        mask=mask,
        depth_m=depth_m,
        intrinsics=intrinsics,
        pose=pose,
        measurement_noise_std_m=0.01,
    )
    assert geom_low_noise.status == GeometryStatus.VALID
    cov_low = geom_low_noise.position_covariance_world

    # Run with noise std 0.05
    geom_high_noise = compute_object_geometry(
        mask=mask,
        depth_m=depth_m,
        intrinsics=intrinsics,
        pose=pose,
        measurement_noise_std_m=0.05,
    )
    assert geom_high_noise.status == GeometryStatus.VALID
    cov_high = geom_high_noise.position_covariance_world

    # Expected difference on diagonal is (0.05^2 - 0.01^2) = 0.0024
    expected_diff = 0.05**2 - 0.01**2
    actual_diff = np.diag(cov_high) - np.diag(cov_low)
    np.testing.assert_allclose(actual_diff, expected_diff, atol=1e-7)


def test_config_resolution_from_arbitrary_cwd(tmp_path):
    """Verify SceneGraphConfig.from_files resolves relative paths regardless of current working directory."""
    original_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        # Relative path from external dir should resolve to repo configs
        config = SceneGraphConfig.from_files("configs/default.yaml")
        assert config is not None
        assert config.perception.mode == "prompt_free"
    finally:
        os.chdir(original_cwd)
