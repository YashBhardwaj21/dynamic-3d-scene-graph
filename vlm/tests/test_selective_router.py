"""Unit tests for Stage 6 Selective Router Policy and Dispatcher."""

import pytest
import sys
from pathlib import Path

# Add project root and vlm to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "vlm"))

from vlm.src.router_policy import evaluate_router_policy, RouterDecision
from vlm.src.selective_router import SelectiveRouter, clean_json_text, parse_vlm_output


def test_gate1_invalid_pose_deferred():
    """Verify that untrustworthy camera pose causes DEFER decision."""
    ev = {
        "entity_id": "chair_001",
        "semantic_class": "chair",
        "has_valid_pose": False,
        "depth_valid": True,
    }
    decision = evaluate_router_policy(ev)
    assert decision.decision == "DEFER"
    assert "UNTRUSTWORTHY_ESTIMATOR_POSE" in decision.reason_codes
    assert not decision.vlm_required


def test_gate1_invalid_depth_on_moving_target():
    """Verify that moving target with invalid depth is deferred."""
    ev = {
        "entity_id": "chair_002",
        "semantic_class": "chair",
        "has_valid_pose": True,
        "depth_valid": False,
        "observed_motion": "MOVING",
        "world_displacement_m": 0.50,
    }
    decision = evaluate_router_policy(ev)
    assert decision.decision == "DEFER"
    assert "INVALID_DEPTH_ON_MOVING_TARGET" in decision.reason_codes


def test_canonical_mobile_agent_resolved_cheaply():
    """Verify person or robot resolves immediately to MOBILE without VLM."""
    for cls_name in ["person", "robot", "quadcopter drone"]:
        ev = {
            "entity_id": f"inst_{cls_name}_001",
            "semantic_class": cls_name,
            "has_valid_pose": True,
            "depth_valid": True,
            "observed_motion": "DYNAMIC",
        }
        decision = evaluate_router_policy(ev)
        assert decision.decision == "RESOLVE_CHEAP"
        assert decision.cheap_label == "MOBILE"
        assert not decision.vlm_required
        assert "CANONICAL_MOBILE_AGENT_PRIOR" in decision.reason_codes


def test_canonical_immovable_resolved_cheaply():
    """Verify door or ceiling fan without displacement resolves to IMMOVABLE without VLM."""
    for cls_name in ["door", "ceiling fan", "wall"]:
        ev = {
            "entity_id": f"inst_{cls_name}_001",
            "semantic_class": cls_name,
            "has_valid_pose": True,
            "depth_valid": True,
            "observed_motion": "STATIC",
            "world_displacement_m": 0.02,
        }
        decision = evaluate_router_policy(ev)
        assert decision.decision == "RESOLVE_CHEAP"
        assert decision.cheap_label == "IMMOVABLE"
        assert not decision.vlm_required
        assert "CANONICAL_IMMOVABLE_ARCHITECTURAL_PRIOR" in decision.reason_codes


def test_stationary_movable_resolved_cheaply():
    """Verify stationary desk or laptop resolves to MOVABLE without VLM."""
    for cls_name in ["desk", "laptop", "backpack"]:
        ev = {
            "entity_id": f"inst_{cls_name}_001",
            "semantic_class": cls_name,
            "has_valid_pose": True,
            "depth_valid": True,
            "observed_motion": "STATIC",
            "world_displacement_m": 0.03,
            "displacement_uncertainty_m": 0.02,
        }
        decision = evaluate_router_policy(ev)
        assert decision.decision == "RESOLVE_CHEAP"
        assert decision.cheap_label == "MOVABLE"
        assert not decision.vlm_required


def test_significant_displacement_escalated_with_images():
    """Verify relocated chair with images escalates to VLM."""
    ev = {
        "entity_id": "inst_chair_002",
        "semantic_class": "chair",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "DISPLACED",
        "world_displacement_m": 0.563,
        "displacement_uncertainty_m": 0.04,
        "has_images": True,
        "views_count": 2,
    }
    decision = evaluate_router_policy(ev)
    assert decision.decision == "ESCALATE"
    assert decision.vlm_required
    assert "SIGNIFICANT_DISPLACEMENT_AFFORDANCE_UNRESOLVED" in decision.reason_codes


def test_significant_displacement_deferred_without_images():
    """Verify relocated chair without images defers instead of failing silently."""
    ev = {
        "entity_id": "inst_chair_002",
        "semantic_class": "chair",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "DISPLACED",
        "world_displacement_m": 0.563,
        "has_images": False,
        "views_count": 0,
    }
    decision = evaluate_router_policy(ev)
    assert decision.decision == "DEFER"
    assert not decision.vlm_required
    assert "MISSING_REQUIRED_VISUAL_CONTEXT" in decision.reason_codes


def test_ambiguous_affordance_escalation():
    """Verify pedestal fan escalates for visual mounting affordance check."""
    ev = {
        "entity_id": "inst_pedestal_fan_001",
        "semantic_class": "pedestal fan",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "STATIC",
        "has_images": True,
        "views_count": 1,
    }
    decision = evaluate_router_policy(ev)
    assert decision.decision == "ESCALATE"
    assert decision.vlm_required
    assert "AMBIGUOUS_PHYSICAL_SUPPORT_AFFORDANCE" in decision.reason_codes


def test_router_output_contract():
    """Verify SelectiveRouter outputs conform to required dictionary schema."""
    router = SelectiveRouter(enable_vlm=False)
    ev = {
        "entity_id": "inst_desk_001",
        "semantic_class": "desk",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "STATIC",
        "world_displacement_m": 0.01,
        "has_images": True,
        "views_count": 3,
    }
    res = router.route_entity(ev)

    required_keys = [
        "entity_id", "decision", "reason_codes", "evidence_valid",
        "evidence_provenance", "vlm_required", "vlm_model", "views_used",
        "prediction", "abstained", "decision_latency_ms", "inference_latency_ms"
    ]
    for k in required_keys:
        assert k in res
    assert res["decision"] == "RESOLVE_CHEAP"
    assert res["prediction"] == "MOVABLE"
    assert not res["abstained"]
    assert res["decision_latency_ms"] < 5.0  # Must be fast sub-millisecond execution
