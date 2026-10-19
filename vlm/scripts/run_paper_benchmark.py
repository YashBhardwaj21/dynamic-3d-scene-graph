#!/usr/bin/env python3
"""Unified Master Orchestrator: Paper-Grade Benchmark Suite.

Executes all 4 experimental axes in sequence:
- Axis 1: Visual Movability Classification & Dual-Model Evaluation
- Axis 2: Estimator-Aware Selective Router Policy Benchmark (B0-B7 & Ablations A1-A4)
- Axis 3: Downstream SLAM Landmark Lifecycle & Scene Graph Fusion
- Axis 4: Statistical Analysis, Wilson CIs, Paired Bootstrap Resampling, Claim-to-Evidence Matrix

Provides:
- Strict two-file ground-truth isolation
- Strict live execution option or reproducible replay execution
- Honest provenance and statistical confidence intervals
- Strictly no unsupported paper claims (ATE/RPE marked NOT_EVALUATED)
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Literal, Optional

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "vlm"))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from vlm.scripts.evaluate_visual_classification import evaluate_visual_classification
from vlm.scripts.evaluate_router_policies import evaluate_router_policies
from vlm.scripts.evaluate_downstream_lifecycle import evaluate_downstream_lifecycle
from vlm.src.statistical_analysis import run_statistical_analysis


def run_paper_benchmark(
    mode: Literal["replay", "live"] = "replay",
    models: List[str] = ["qwen2_5_vl_3b"],
    view_modes: List[str] = ["CROP_ONLY", "CROP_PLUS_CONTEXT"],
    replay_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    skip_offline: bool = True,
) -> Dict[str, Any]:
    """Execute complete paper-grade benchmark across all experimental axes."""
    t0 = time.time()
    run_id = f"paper_benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if output_dir is None:
        output_dir = repo_root / "vlm" / "outputs" / "benchmark_runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MASTER ORCHESTRATOR: ESTIMATOR-AWARE SELECTIVE VLM BENCHMARK")
    print(f"Run ID: {run_id}")
    print(f"Execution Mode: {mode.upper()}")
    print(f"Target Models: {models}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)

    # AXIS 1: Visual Classification
    if mode == "live":
        print("\n[STEP 1/4] Executing Axis 1: Live Visual Classification...")
        vis_summary = evaluate_visual_classification(
            model_keys=models,
            view_modes=view_modes,
            run_dir=output_dir,
            skip_offline_models=skip_offline,
        )
        replay_source_dir = output_dir
    else:
        print("\n[STEP 1/4] Axis 1: Using recorded predictions from prior run...")
        if replay_dir is None:
            # Find latest visual_eval run
            bench_runs = repo_root / "vlm" / "outputs" / "benchmark_runs"
            vis_runs = sorted(bench_runs.glob("visual_eval_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not vis_runs:
                raise FileNotFoundError("Replay mode requested but no prior visual_eval_* run found.")
            replay_source_dir = vis_runs[0]
        else:
            replay_source_dir = replay_dir
        print(f"  -> Replay source: {replay_source_dir.name}")

    # AXIS 2: Router Policy Suite
    print("\n[STEP 2/4] Executing Axis 2: Router Policy Suite & Comparative Baselines...")
    router_summary = evaluate_router_policies(
        model_key=models[0],
        execution_mode="replay",
        replay_run_dir=replay_source_dir,
        run_dir=output_dir,
    )

    # AXIS 3: Downstream SLAM Landmark Lifecycle
    print("\n[STEP 3/4] Executing Axis 3: Downstream SLAM Landmark Lifecycle...")
    lifecycle_summary = evaluate_downstream_lifecycle(
        router_run_dir=output_dir,
        output_run_dir=output_dir,
    )

    # AXIS 4: Statistical Analysis & Claim-to-Evidence Matrix
    print("\n[STEP 4/4] Executing Axis 4: Statistical Analysis & Claim Verification...")
    stats_summary = run_statistical_analysis(
        router_run_dir=output_dir,
        output_dir=output_dir,
    )

    elapsed_s = time.time() - t0
    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED SUCCESSFULLY in {elapsed_s:.1f}s")
    print(f"Master Output Run Directory: {output_dir}")
    print("Key Artifacts Generated:")
    print(f"  • {output_dir / 'router_policies_summary.json'}")
    print(f"  • {output_dir / 'downstream_lifecycle_summary.json'}")
    print(f"  • {output_dir / 'statistical_metrics_summary.json'}")
    print(f"  • {output_dir / 'claim_to_evidence_matrix.md'}")
    print(f"  • {output_dir / 'paper_benchmark_report.md'}")
    print("=" * 80)

    return {
        "run_id": run_id,
        "elapsed_seconds": round(elapsed_s, 2),
        "output_dir": str(output_dir.relative_to(repo_root.resolve())).replace("\\", "/"),
        "router_summary": router_summary,
        "lifecycle_summary": lifecycle_summary,
        "statistical_summary": stats_summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete paper-grade benchmark orchestrator.")
    parser.add_argument("--mode", default="replay", choices=["replay", "live"])
    parser.add_argument("--models", nargs="+", default=["qwen2_5_vl_3b"])
    parser.add_argument("--replay-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-offline", action="store_true")
    args = parser.parse_args()

    run_paper_benchmark(
        mode=args.mode,
        models=args.models,
        replay_dir=args.replay_dir,
        output_dir=args.output_dir,
        skip_offline=args.skip_offline,
    )
