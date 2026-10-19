"""Unit and Regression Tests for Paper Benchmark Two-File Isolation and Image Integrity.

Verifies:
1. Strict two-file isolation: paper_benchmark_inputs.json contains ZERO ground truth or decision fields.
2. paper_benchmark_ground_truth.json correctly matches physical instance IDs.
3. All image files exist, decode cleanly with PIL, and SHA-256 hashes match.
4. Model registry specifies complete configurations for both qwen2_5_vl_3b and gemma3_4b.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
from PIL import Image

repo_root = Path(__file__).resolve().parent.parent.parent


def hash_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def test_two_file_manifest_isolation():
    """Verify inference inputs manifest contains ZERO evaluator labels or decision targets."""
    inputs_path = repo_root / "vlm" / "benchmarks" / "paper_benchmark_inputs.json"
    gt_path = repo_root / "vlm" / "benchmarks" / "paper_benchmark_ground_truth.json"

    assert inputs_path.exists(), f"Missing inputs manifest at {inputs_path}"
    assert gt_path.exists(), f"Missing ground truth manifest at {gt_path}"

    with open(inputs_path, "r", encoding="utf-8") as f:
        inputs_data = json.load(f)["instances"]

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)["labels"]

    forbidden_fields = [
        "contextual_movability_gt",
        "current_motion_gt",
        "displacement_m_gt",
        "slam_landmark_validity_gt",
        "expected_decision",
        "expected_reason",
        "annotation_provenance",
        "adjudication_status",
    ]

    for rec in inputs_data:
        eid = rec["physical_instance_id"]
        # Ground truth must exist in evaluator file
        assert eid in gt_data, f"Physical instance {eid} missing from ground truth file"

        # Forbidden fields MUST NOT exist in inference inputs file
        for f in forbidden_fields:
            assert f not in rec, f"Leaked ground truth field '{f}' found in inference input for {eid}"
            if "visual_inputs" in rec:
                assert f not in rec["visual_inputs"]

        # Evaluator ground truth must have valid classes
        gt_rec = gt_data[eid]
        assert gt_rec["contextual_movability_gt"] in ["IMMOVABLE", "MOVABLE", "MOBILE"]
        assert gt_rec["current_motion_gt"] in ["STATIC", "DISPLACED", "DYNAMIC", "UNKNOWN"]
        assert gt_rec["slam_landmark_validity_gt"] in ["RETAIN_LANDMARK", "INVALIDATE_RELOCATED", "INVALIDATE_DYNAMIC_AGENT"]


def test_image_integrity_and_hashes():
    """Verify all referenced images in the benchmark exist, decode cleanly, and match SHA-256 hashes."""
    inputs_path = repo_root / "vlm" / "benchmarks" / "paper_benchmark_inputs.json"
    with open(inputs_path, "r", encoding="utf-8") as f:
        instances = json.load(f)["instances"]

    image_eligible_count = 0
    for rec in instances:
        vis = rec.get("visual_inputs", {})
        if not vis.get("has_images"):
            continue

        image_eligible_count += 1
        crop_paths = vis.get("crop_view_paths", [])
        scene_paths = vis.get("scene_view_paths", [])
        expected_hashes = vis.get("image_sha256", [])

        assert len(crop_paths) == len(scene_paths)
        assert len(expected_hashes) == len(crop_paths) + len(scene_paths)

        hash_idx = 0
        for c_rel, s_rel in zip(crop_paths, scene_paths):
            c_p = repo_root / c_rel
            s_p = repo_root / s_rel

            assert c_p.exists(), f"Missing crop image: {c_p}"
            assert s_p.exists(), f"Missing scene image: {s_p}"

            # Verify PIL decoding
            with Image.open(c_p) as im:
                im.verify()
            with Image.open(s_p) as im:
                im.verify()

            # Verify content SHA-256 hash
            c_hash = hash_file_sha256(c_p)
            s_hash = hash_file_sha256(s_p)
            assert c_hash == expected_hashes[hash_idx]
            assert s_hash == expected_hashes[hash_idx + 1]
            hash_idx += 2

    assert image_eligible_count == 14, f"Expected 14 image-eligible instances, found {image_eligible_count}"


def test_model_registry_schema():
    """Verify model_registry.json contains required entries and parameters for Qwen and Gemma."""
    reg_path = repo_root / "vlm" / "benchmarks" / "model_registry.json"
    assert reg_path.exists()

    with open(reg_path, "r", encoding="utf-8") as f:
        registry = json.load(f)["models"]

    for mkey in ["qwen2_5_vl_3b", "gemma3_4b"]:
        assert mkey in registry, f"Missing model {mkey} in registry"
        m = registry[mkey]
        for field in ["model_identifier", "server_url", "generation_parameters", "image_input_specification"]:
            assert field in m, f"Model {mkey} missing field {field}"
        assert m["generation_parameters"]["temperature"] == 0.0
        assert m["generation_parameters"]["max_tokens"] >= 150
