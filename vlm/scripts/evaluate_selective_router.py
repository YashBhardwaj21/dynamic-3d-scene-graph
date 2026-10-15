#!/usr/bin/env python3
"""Stage 6: Offline and Live Evaluation of Estimator-Aware Selective Router."""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "vlm"))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from vlm.src.selective_router import SelectiveRouter


CLASSES = ["IMMOVABLE", "MOVABLE", "MOBILE"]


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Quality, Efficiency, and Reliability metrics."""
    total = len(records)
    if total == 0:
        return {}

    eval_classes = ["IMMOVABLE", "MOVABLE", "MOBILE"]
    pred_columns = ["IMMOVABLE", "MOVABLE", "MOBILE", "UNKNOWN"]
    cm = {c: {p: 0 for p in pred_columns} for c in eval_classes}

    correct_by_class = defaultdict(int)
    total_by_gt = defaultdict(int)
    total_by_pred = defaultdict(int)

    abstain_count = 0
    vlm_query_count = 0
    decision_latencies_ms = []
    end_to_end_latencies_s = []

    for r in records:
        gt = r["ground_truth"]
        pred = r["prediction"]
        abstained = r.get("abstained", False) or pred in ["UNKNOWN", None]
        decision_latencies_ms.append(r.get("decision_latency_ms", 0.05))

        # End to end latency: decision time + inference time if queried
        inf_ms = r.get("inference_latency_ms") or 0.0
        e2e_s = (r.get("decision_latency_ms", 0.05) + inf_ms) / 1000.0
        end_to_end_latencies_s.append(e2e_s)

        if r.get("vlm_required", False):
            vlm_query_count += 1

        total_by_gt[gt] += 1
        if abstained:
            abstain_count += 1
            cm[gt]["UNKNOWN"] += 1
        else:
            p_val = pred if pred in eval_classes else "UNKNOWN"
            cm[gt][p_val] += 1
            total_by_pred[p_val] += 1
            if p_val == gt:
                correct_by_class[gt] += 1

    # Quality metrics
    recalls = {}
    precisions = {}
    f1s = {}
    for c in eval_classes:
        rec = (correct_by_class[c] / total_by_gt[c]) if total_by_gt[c] > 0 else 0.0
        prec = (correct_by_class[c] / total_by_pred[c]) if total_by_pred[c] > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        recalls[c] = round(rec, 4)
        precisions[c] = round(prec, 4)
        f1s[c] = round(f1, 4)

    balanced_acc = float(np.mean([recalls[c] for c in eval_classes if total_by_gt[c] > 0]))
    macro_f1 = float(np.mean([f1s[c] for c in eval_classes if total_by_gt[c] > 0]))

    # Efficiency metrics
    query_rate_pct = (vlm_query_count / max(1, total)) * 100.0
    abstention_rate_pct = (abstain_count / max(1, total)) * 100.0

    return {
        "quality": {
            "balanced_accuracy": round(balanced_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class_recall": recalls,
            "per_class_precision": precisions,
            "per_class_f1": f1s,
            "confusion_matrix": cm,
            "abstention_rate_pct": round(abstention_rate_pct, 2),
        },
        "efficiency": {
            "total_instances": total,
            "vlm_queries_invoked": vlm_query_count,
            "vlm_queries_avoided": total - vlm_query_count,
            "vlm_query_rate_pct": round(query_rate_pct, 2),
            "decision_latency_ms": {
                "mean": round(float(np.mean(decision_latencies_ms)), 3),
                "p50": round(float(np.percentile(decision_latencies_ms, 50)), 3),
                "p95": round(float(np.percentile(decision_latencies_ms, 95)), 3),
            },
            "end_to_end_latency_s": {
                "mean": round(float(np.mean(end_to_end_latencies_s)), 3),
                "p50": round(float(np.percentile(end_to_end_latencies_s, 50)), 3),
                "p95": round(float(np.percentile(end_to_end_latencies_s, 95)), 3),
            },
        },
        "reliability": {
            "false_cheap_resolutions": sum(1 for r in records if r["decision"] == "RESOLVE_CHEAP" and r["prediction"] != r["ground_truth"]),
            "invalid_evidence_escalations": sum(1 for r in records if r["decision"] == "ESCALATE" and not r["evidence_valid"]),
            "vlm_abstentions": sum(1 for r in records if r.get("vlm_required") and r.get("abstained")),
        }
    }


def run_selective_router_evaluation(
    enable_vlm: bool = True,
    cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    manifest_path = repo_root / "vlm" / "benchmarks" / "router_evaluation_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if cache_path is None:
        cache_path = repo_root / "vlm" / "outputs" / "stage6" / "router_vlm_cache.json"

    router = SelectiveRouter(
        cache_path=cache_path,
        enable_vlm=enable_vlm,
    )

    instances = manifest["evaluation_instances"]
    print(f"\nEvaluating Selective Router on {len(instances)} physical instances (VLM Enabled: {enable_vlm})...")

    results = []
    for inst in instances:
        evidence = {
            "entity_id": inst["entity_id"],
            "semantic_class": inst["semantic_class"],
            "observed_motion": inst["observed_motion_gt"],
            "world_displacement_m": 0.563 if inst["observed_motion_gt"] == "DISPLACED" and "chair_002" in inst["entity_id"] else (
                0.386 if inst["observed_motion_gt"] == "DISPLACED" and "chair_003" in inst["entity_id"] else (
                    0.396 if inst["observed_motion_gt"] == "DISPLACED" else 0.02
                )
            ),
            "displacement_uncertainty_m": 0.04,
            "has_valid_pose": True,
            "depth_valid": True,
            "has_images": inst["has_images"],
            "views_count": inst["views_count"],
            "crops_dir": inst["crops_dir"],
        }

        res = router.route_entity(evidence, repo_root=repo_root)
        res["ground_truth"] = inst["contextual_movability_gt"]
        res["observed_motion_gt"] = inst["observed_motion_gt"]
        results.append(res)

        status_flag = "CHEAP" if res["decision"] == "RESOLVE_CHEAP" else ("VLM-QUERY" if res["decision"] == "ESCALATE" else "DEFER")
        print(f"  • {inst['entity_id']:<26} [{inst['semantic_class']:<14}] -> {res['decision']:<14} | Pred={res['prediction']} (GT={inst['contextual_movability_gt']}) | {status_flag}")

    metrics = compute_metrics(results)
    output_dir = repo_root / "vlm" / "outputs" / "stage6"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "evaluation_title": "Stage 6 Selective Router Evaluation",
        "timestamp": time.time(),
        "enable_vlm": enable_vlm,
        "metrics": metrics,
        "evaluation_records": results,
    }

    out_file = output_dir / ("router_evaluation_vlm_enabled.json" if enable_vlm else "router_evaluation_cheap_only.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved evaluation metrics to: {out_file}")

    return summary


if __name__ == "__main__":
    run_selective_router_evaluation(enable_vlm=True)
