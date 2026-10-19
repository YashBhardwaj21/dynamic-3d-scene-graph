# Chronological Experiment Execution Log

## 1. Overview & Protocol Standard

This document maintains a chronological, immutable log of all validated experimental executions conducted during the Master Engineering Task for the VLM subsystem of `Dynamic 3D Scene Graph`.

Every experiment record specifies:
- Run Identifier
- Execution Timestamp & Environment
- Git Revision SHA
- Command Executed
- Input Population & Configuration
- Output Deliverables & SHA-256 Hashes
- Verified Results & Key Metrics

---

## 2. Chronological Experiment Records

### EXP-01: Stage 5 Evidence Pipeline Audit & Coverage Verification
- **Run ID:** `EXP-20261015-STAGE5-EVIDENCE`
- **Timestamp:** 2026-10-15 19:42:00 UTC
- **Git Commit:** `e9c336408479b2a2a468882974a3a022a48ab0e7`
- **Command:** `py -3 vlm/scripts/evaluate_stage5_evidence.py`
- **Configuration:** 7,956 observations from `vlm/outputs/object_observations.jsonl`.
- **Output Artifacts:** `vlm/outputs/stage5_evidence_metrics.json`
- **Verified Metrics:**
  - Total records processed: `7,956`
  - Matched support observations: `4,421` ($55.57\%$ coverage)
  - Unmapped out-of-vocabulary records: `3,535` ($44.43\%$)
  - Matched Support Macro-F1: `99.76%` (with prominent coverage caveat)
  - Track fragmentation rate: `72.78%` (1,008 tracks $\le 4$ frames)
  - Mean relative displacement error across 3 relocations: `30.42%` (MAE = `0.1247 m`)

---

### EXP-02: Rüdt-Style Movability Baseline Evaluation
- **Run ID:** `EXP-20261015-RUDT-BASELINES`
- **Timestamp:** 2026-10-15 20:12:00 UTC
- **Git Commit:** `e9c336408479b2a2a468882974a3a022a48ab0e7`
- **Command:** `py -3 vlm/scripts/evaluate_rudt_movability_baseline.py`
- **Configuration:** Evaluated on all 39 physical instances with verified crops from `vlm/outputs/baseline_checkpoint.json`.
- **Output Artifacts:**
  - `vlm/outputs/movability_baseline_results.json`
  - `vlm/outputs/movability_baseline_report.md`
- **Verified Metrics:**
  - M0 (Semantic Class Prior): Balanced Acc = `100.00%`, Macro-F1 = `100.00%`, Latency = `0.00 s` (on 39 prototype objects)
  - M1 (Single-View VLM): Balanced Acc = `48.65%`, Macro-F1 = `48.00%`, Latency p50 = `16.05 s`
  - M2 (Multi-View VLM): Balanced Acc = `48.65%`, Macro-F1 = `48.00%`, Latency p50 = `24.19 s`
  - M3 (Structured Alternate Prompt): Balanced Acc = `50.00%`, Macro-F1 = `48.68%`, Latency p50 = `23.22 s`
- **Key Insight:** Standalone VLMs persistently misclassify architectural structures (`door`, `ceiling fan`) as `MOVABLE` due to linguistic action affordances.

---

### EXP-03: Stage 6 Decoupled Router Policy Ablation Matrix
- **Run ID:** `EXP-20261015-STAGE6-ROUTER-ABLATION`
- **Timestamp:** 2026-10-15 20:09:48 UTC
- **Git Commit:** `e9c336408479b2a2a468882974a3a022a48ab0e7`
- **Command:** `py -3 vlm/scripts/run_stage6_ablation.py`
- **Configuration:** Evaluated policies R0, R1, R2, R3, R4, Random (1,000 seeds), and Oracle on the 15-instance balanced manifest with dual views (View A: N=11 image-eligible; View B: N=15 end-to-end system).
- **Output Artifacts:**
  - `vlm/outputs/stage6/router_ablation_results.json`
  - `vlm/outputs/stage6/router_evaluation_report.md`
- **Verified Metrics (View B End-to-End System):**
  - R0 (Semantic Prior): Balanced Acc = `85.2%`, Query Rate = `0.0%`, Latency = `0.00 s`
  - R1 (Geometry/Motion Only): Balanced Acc = `74.1%`, Query Rate = `0.0%`, Latency = `0.00 s`
  - R2 (Non-Estimator-Aware Selective): Balanced Acc = `77.8%`, Query Rate = `33.3%`, Latency = `10.00 s`
  - R3 (Estimator-Aware Selective): Balanced Acc = `77.8%`, Query Rate = `33.3%`, Latency = `10.00 s`
  - R4 (All-Eligible VLM Reference): Balanced Acc = `33.3%`, Query Rate = `73.3%`, Latency = `19.31 s`
  - Random Query Baseline (1,000 seeds): Mean Balanced Acc = `89.7% (+/- 7.3%)`, Query Rate = `33.3%`
  - Oracle Diagnostic Upper Bound: Balanced Acc = `77.8%`, Query Rate = `33.3%`
- **Counterexample Suite Evaluation:**
  - R0 Naive Prior: `30.0%` (3/10 correct)
  - R3 Contextual Selective: `90.0%` (9/10 correct)
  - Contextual Accuracy Gain: `+60.0%`

---

### EXP-04: Phase 6 Controlled VLM Ablation Study
- **Run ID:** `EXP-20261015-PHASE6-CONTROLLED-ABLATIONS`
- **Timestamp:** 2026-10-15 20:13:27 UTC
- **Git Commit:** `e9c336408479b2a2a468882974a3a022a48ab0e7`
- **Command:** `py -3 vlm/scripts/run_vlm_controlled_ablations.py`
- **Configuration:** Matched prompt, view-count (matched 3-view N=8, matched 2-view N=24), and input representation ablations on all 39 image-eligible instances.
- **Output Artifacts:**
  - `vlm/outputs/vlm_controlled_ablations.json`
  - `vlm/outputs/vlm_controlled_ablations_report.md`
- **Verified Metrics:**
  - Prompt Ablation (P1/P2/P3): `50.0%` / `48.6%` / `48.6%`
  - Matched 3-Views (1v/2v/3v): `100.0%` / `100.0%` / `100.0%` (Token consumption: 331.5 -> 420.0 -> 564.4 tokens; Latency: 15.87s -> 30.78s)
  - Input Representation (Crop/Single/Multi): `48.6%` / `48.6%` / `50.0%`
  - Key finding: Extra views do not fix linguistic mobility bias and yield diminishing returns on indoor scenes.

---

### EXP-05: Multi-Session Pipeline Orchestration & Scene Graph Fusion
- **Run ID:** `EXP-20261015-FULL-PIPELINE-FUSION`
- **Timestamp:** 2026-10-15 20:14:15 UTC
- **Git Commit:** `e9c336408479b2a2a468882974a3a022a48ab0e7`
- **Command:** `py -3 vlm/scripts/run_full_experiment.py`
- **Configuration:** Complete run across 7,956 observations from Session 1 and Session 2.
- **Output Artifacts:**
  - `vlm/outputs/full_experiment_run/scene_graph.json`
  - `vlm/outputs/full_experiment_run/association_baselines_summary.json`
  - `vlm/outputs/full_experiment_run/run_manifest.json`
- **Verified Metrics:**
  - Total persistent scene graph nodes: `195`
  - S1 clusters: `102`; S2 clusters: `166`
  - Matched cross-session pairs: `41`
  - Stationary merged entities: `29`
  - Relocated fused entities: `12`
  - Invalidated SLAM landmarks: `12`
  - Execution time: `6.99 s` (1,138 records/sec)

---

### EXP-06: Downstream SLAM & Dynamic Scene Graph Validation
- **Run ID:** `EXP-20261015-DOWNSTREAM-SLAM-EFFECTS`
- **Timestamp:** 2026-10-16 02:37:06 UTC
- **Git Commit:** `main` (clean working tree modifications)
- **Command:** `py -3 vlm/scripts/evaluate_downstream_slam_effects.py`
- **Configuration:** Evaluated landmark retention, false invalidations, and tracking telemetry on `scene_graph.json` and `my_desk_sequence`.
- **Output Artifacts:**
  - `vlm/outputs/downstream_slam_effects.json`
  - `vlm/outputs/downstream_slam_effects_report.md`
- **Verified Metrics:**
  - Correct Landmark Retention Rate: `91.1%` (41/45 stationary objects retained)
  - False Invalidation Rate: `8.9%` (4/45)
  - SLAM Tracking Ratio: `88.75%` (71/80 valid poses on `my_desk_sequence`, 0 tracking losses)
  - SLAM Trajectory ATE/RPE: `null` / `UNEVALUATED_NO_MOCAP_GROUND_TRUTH` (No continuous 6-DoF optical MoCap recorded for room sequence)
  - VLM Compute Reduction: `72.7%` queries avoided vs All-VLM (3 queries in R3 vs 11 in R4)
  - Latency Speedup: `2.92x`

---

### EXP-07: Full Unit & Regression Suite Execution
- **Run ID:** `EXP-20261016-FULL-REGRESSION-SUITE`
- **Timestamp:** 2026-10-16 02:52:45 UTC
- **Git Commit:** `main`
- **Command:** `py -3 -m pytest vlm/tests -q`
- **Configuration:** 13 test files covering input validation, operational utility thresholding, zero-inlier handling, and fail-closed contracts.
- **Result:** **115 passed / 0 failed / 0 skipped in 3.45s**

---

### EXP-08: Audited Expanded Benchmark & Publication Package
- **Run ID:** `EXP-20261019-AUDITED-PAPER-BENCHMARK`
- **Timestamp:** 2026-10-19 11:45:00 UTC
- **Branch:** `research-vlm` (commit `7a71ddc...` base, immutable historical pilot preserved)
- **Commands Executed:**
  - `python vlm/scripts/audit_benchmark_population.py`
  - `python vlm/scripts/run_paper_benchmark.py --protocol visual-live --models qwen2_5_vl_3b ...`
  - `python vlm/scripts/run_paper_benchmark.py --protocol policy-replay --models qwen2_5_vl_3b ...`
  - `python vlm/scripts/run_paper_benchmark.py --protocol router-live --models qwen2_5_vl_3b ...`
  - `python vlm/scripts/generate_model_comparison_report.py ...`
  - `python vlm/scripts/validate_final_reports.py ...`
  - `python -m pytest vlm/tests/ -q`
- **Key Verified Metrics:**
  - Audited Population: 53 catalog entities, 42 unique image-eligible instances with verified SHA-256 hashes.
  - Qwen2.5-VL-3B: Overall Accuracy `95.2%` (40/42, 95% Wilson CI `[84.2%, 98.7%]`), Balanced Accuracy `98.2%`, Macro-F1 `90.7%`, Median Latency `23.5 s`.
  - B6 Estimator-Aware Router: Downstream landmark action accuracy `94.3%`, Landmark retention recall `100.0%`, False invalidation rate `0.0%`.
  - Trajectory ATE/RPE: Explicitly recorded as `NOT_EVALUATED`.
  - Unit/Regression Test Suite: **233 passed / 0 failed in 78.33s**.
- **Deliverables Directory:** [`vlm/outputs/final_report/`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/outputs/final_report/)

