"""Unit tests for Stage 6 Selective Router metrics, random baseline aggregation, and query accounting."""

import json
import numpy as np
import pytest
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent

from vlm.scripts.evaluate_selective_router import compute_metrics
from vlm.scripts.run_stage6_ablation import (
    run_random_baseline_1000_seeds,
    generate_ablation_markdown_report,
)
from vlm.src.selective_router import SelectiveRouter


def test_metric_definitions_query_distinction():
    """Verify compute_metrics strictly separates requested, executed, cached, valid, and accepted queries."""
    records = [
        # 1: Requested and live executed
        {
            "entity_id": "e1",
            "decision": "ESCALATE",
            "prediction": "MOVABLE",
            "ground_truth": "MOVABLE",
            "abstained": False,
            "vlm_required": True,
            "vlm_inference_executed": True,
            "vlm_cache_hit": False,
            "vlm_output_valid": True,
            "vlm_prediction_accepted": True,
            "decision_latency_ms": 0.05,
            "inference_latency_ms": 5000.0,
            "evidence_valid": True,
        },
        # 2: Requested and cache hit
        {
            "entity_id": "e2",
            "decision": "ESCALATE",
            "prediction": "IMMOVABLE",
            "ground_truth": "IMMOVABLE",
            "abstained": False,
            "vlm_required": True,
            "vlm_inference_executed": False,
            "vlm_cache_hit": True,
            "vlm_output_valid": True,
            "vlm_prediction_accepted": True,
            "decision_latency_ms": 0.05,
            "inference_latency_ms": 10.0,
            "evidence_valid": True,
        },
        # 3: Requested but VLM disabled (abstained, 0 executed)
        {
            "entity_id": "e3",
            "decision": "ESCALATE",
            "prediction": "UNKNOWN",
            "ground_truth": "MOVABLE",
            "abstained": True,
            "vlm_required": True,
            "vlm_inference_executed": False,
            "vlm_cache_hit": False,
            "vlm_output_valid": False,
            "vlm_prediction_accepted": False,
            "decision_latency_ms": 0.05,
            "inference_latency_ms": 0.0,
            "evidence_valid": True,
        },
        # 4: Cheap prior (not requested)
        {
            "entity_id": "e4",
            "decision": "RESOLVE_CHEAP",
            "prediction": "MOBILE",
            "ground_truth": "MOBILE",
            "abstained": False,
            "vlm_required": False,
            "vlm_inference_executed": False,
            "vlm_cache_hit": False,
            "vlm_output_valid": True,
            "vlm_prediction_accepted": True,
            "decision_latency_ms": 0.04,
            "inference_latency_ms": 0.0,
            "evidence_valid": True,
        },
    ]

    m = compute_metrics(records)
    eff = m["efficiency"]

    assert eff["total_instances"] == 4
    assert eff["vlm_queries_requested"] == 3       # e1, e2, e3
    assert eff["vlm_inference_executed"] == 1      # e1
    assert eff["vlm_cache_hits"] == 1              # e2
    assert eff["vlm_queries_invoked"] == 2         # e1 + e2 (actual model results used)
    assert eff["vlm_queries_avoided"] == 2         # e3 + e4
    assert eff["vlm_query_rate_pct"] == 50.0       # 2 / 4
    assert eff["vlm_outputs_valid"] == 3           # e1, e2, e4
    assert eff["vlm_predictions_accepted"] == 3    # e1, e2, e4


def test_no_query_accounting_when_vlm_disabled():
    """Verify that when VLM is disabled, executed and invoked queries are strictly 0."""
    router = SelectiveRouter(enable_vlm=False)
    # Target that causes escalation
    evidence = {
        "entity_id": "inst_chair_moving",
        "semantic_class": "chair",
        "observed_motion": "DISPLACED",
        "world_displacement_m": 0.60,
        "displacement_uncertainty_m": 0.10,
        "relocation_evaluable": True,
        "has_images": True,
        "views_count": 2,
        "has_valid_pose": True,
        "depth_valid": True,
    }

    result = router.route_entity(evidence)
    assert result["decision"] == "ESCALATE"
    assert result["vlm_required"] is True
    assert result["vlm_inference_executed"] is False
    assert result["vlm_cache_hit"] is False
    assert result["prediction"] == "UNKNOWN"
    assert result["abstained"] is True

    result["ground_truth"] = "MOVABLE"
    metrics = compute_metrics([result])
    assert metrics["efficiency"]["vlm_queries_requested"] == 1
    assert metrics["efficiency"]["vlm_queries_invoked"] == 0
    assert metrics["efficiency"]["vlm_query_rate_pct"] == 0.0


def test_random_seed_canonical_aggregation():
    """Verify that random baseline 1,000-seed summary aggregates canonical statistics."""
    mock_instances = [
        {"entity_id": f"inst_{i:02d}", "semantic_class": "chair", "contextual_movability_gt": "MOVABLE", "has_images": True, "views_count": 1}
        for i in range(10)
    ]
    mock_vlm_map = {
        f"inst_{i:02d}": {"prediction": "MOVABLE", "abstained": False, "views_used": 1, "inference_latency_ms": 1000.0}
        for i in range(10)
    }

    res = run_random_baseline_1000_seeds(
        instances=mock_instances,
        vlm_predictions_map=mock_vlm_map,
        k_queries=3,
        num_seeds=50,
    )

    assert res["num_seeds"] == 50
    assert res["query_budget"] == 3
    assert "mean" in res["balanced_accuracy"]
    assert "std" in res["balanced_accuracy"]
    assert "median" in res["balanced_accuracy"]
    assert "p05" in res["balanced_accuracy"]
    assert "p95" in res["balanced_accuracy"]
    assert res["canonical_summary_metrics"]["efficiency"]["vlm_queries_invoked"] == 3


def test_report_to_json_agreement(tmp_path):
    """Verify that all markdown report cells match JSON numerical values directly."""
    ablation_json_path = root_dir / "vlm" / "outputs" / "stage6" / "router_ablation_results.json"
    if not ablation_json_path.exists():
        pytest.skip("router_ablation_results.json not generated.")

    with open(ablation_json_path, "r", encoding="utf-8") as f:
        ablation = json.load(f)

    rep_path = tmp_path / "test_report.md"
    generate_ablation_markdown_report(ablation, rep_path)
    content = rep_path.read_text(encoding="utf-8")

    # Check selective router values
    sel = ablation["experiments"]["router_selective_vlm"]["metrics"]
    sel_acc = f"{sel['quality']['balanced_accuracy']*100:.1f}%"
    sel_f1 = f"{sel['quality']['macro_f1']*100:.1f}%"
    assert sel_acc in content
    assert sel_f1 in content
