# Reproducibility Guide: VLM Subsystem & Evaluation Pipeline

## 1. System Requirements & Environment Specifications

The experimental pipeline is designed for fully deterministic, offline reproducibility without requiring external cloud API keys, paid credits, or specialized GPU clusters.

- **Operating System:** Windows 10/11, Ubuntu 20.04/22.04 LTS, or WSL2.
- **Python Version:** Python 3.10 to 3.13 (verified on Python 3.13.7 64-bit).
- **Core Dependencies:**
  - `numpy >= 1.24.0`
  - `Pillow >= 9.5.0`
  - `pytest >= 7.3.0`
- **Optional Local Inference Server (for Live Mode only):**
  - `llama-server` or `vLLM` serving `Qwen2.5-VL-3B-Instruct` (GGUF Q4_K_M) on `http://127.0.0.1:8089/v1/chat/completions`.
  - Replay mode evaluates cached inferences directly without network access.

---

## 2. Step-by-Step Reproduction Protocol

Execute the following sequential commands from the repository root:

### Step 1: Execute Full Unit & Regression Test Suite
Runs all 113 unit tests across 13 test modules covering visual input contracts, schema parsing, decoupled routing policies, adapter zero-leakage, and report consistency:
```bash
py -3 -m pytest vlm/tests -v
```
*Expected Result:* `113 passed in ~4.5s`.

---

### Step 2: Generate Authoritative Canonical Manifest
Scans all 52 physical ground-truth entities, 39 multi-view image crop directories, computes cryptographic SHA-256 hashes for all JPEG assets, indexes split memberships, and compiles the 10-case counterexample suite:
```bash
py -3 vlm/scripts/generate_canonical_manifest.py
```
*Output Deliverable:* `vlm/benchmarks/canonical_benchmark_manifest.json`

---

### Step 3: Run Stage 6 Decoupled Router Ablation Matrix
Evaluates policies R0, R1, R2, R3, R4, the 1,000-seed Monte-Carlo random baseline, the oracle diagnostic upper bound, and the counterexample suite under dual evaluation views (View A: Matched Image-Eligible; View B: End-to-End System):
```bash
py -3 vlm/scripts/run_stage6_ablation.py
```
*Output Deliverables:*
- `vlm/outputs/stage6/router_ablation_results.json`
- `vlm/outputs/stage6/router_evaluation_report.md`

*Key Target Metrics (View B):*
- R0 Balanced Accuracy: `85.2%` (Query Rate: `0.0%`)
- R1 Balanced Accuracy: `74.1%` (Query Rate: `0.0%`)
- R3 Balanced Accuracy: `77.8%` (Query Rate: `33.3%`)
- R4 Balanced Accuracy: `33.3%` (Query Rate: `73.3%`)
- Random Query Baseline (1,000 seeds): `89.7% (+/- 7.3%)`
- Counterexample Suite Accuracy: R0 = `30.0%` vs R3 = `90.0%` (`+60.0%` gain)

---

### Step 4: Run Phase 6 Controlled VLM Ablation Study
Executes matched prompt formulation ablations (P1 vs P2 vs P3), view-count ablations on strictly matched subsets (matched 3-view N=8, matched 2-view N=24), input representation ablations, and failure taxonomy compilation:
```bash
py -3 vlm/scripts/run_vlm_controlled_ablations.py
```
*Output Deliverables:*
- `vlm/outputs/vlm_controlled_ablations.json`
- `vlm/outputs/vlm_controlled_ablations_report.md`

---

### Step 5: Execute Complete Multi-Session Pipeline Orchestration
Executes the full pipeline from 7,956 multi-session observations through Stage 5 evidence extraction, SE(2) registration, cross-session candidate matching, and contextual scene graph fusion:
```bash
py -3 vlm/scripts/run_full_experiment.py
```
*Output Deliverables:*
- `vlm/outputs/full_experiment_run/scene_graph.json` (195 persistent 3D nodes)
- `vlm/outputs/full_experiment_run/association_baselines_summary.json`
- `vlm/outputs/full_experiment_run/per_instance_association_diagnostics.json`
- `vlm/outputs/full_experiment_run/run_manifest.json`

---

### Step 6: Evaluate Downstream SLAM Effects & Landmark Reliability
Directly evaluates downstream metric SLAM landmark retention, false invalidation, trajectory error, and compute reduction:
```bash
# Evaluate benchmark trajectory error (TUM fr1 desk)
py -3 tools/evaluation/evaluate_trajectory.py --sequence tum_fr1_desk --estimated_tum archive/historical_results/tum_fr1_desk/estimated.tum --groundtruth archive/historical_results/tum_fr1_desk/groundtruth.tum --output_dir vlm/outputs/downstream_slam_evaluation

# Evaluate downstream landmark reliability and compute avoidance
py -3 vlm/scripts/evaluate_downstream_slam_effects.py
```
*Output Deliverables:*
- `vlm/outputs/downstream_slam_effects.json`
- `vlm/outputs/downstream_slam_effects_report.md`

*Key Target Metrics:*
- Total Scene Graph Nodes: `195`
- Correct Landmark Retention Rate: `91.1%` (41 / 45 stationary entities)
- False Invalidation Rate: `8.9%` (4 / 45)
- SLAM ATE RMSE: `0.0545 m` (`5.45 cm`)
- SLAM RPE RMSE: `0.0349 m` (`3.49 cm`)
- VLM Compute Reduction: `54.5%` queries avoided vs All-VLM

---

## 3. Verification & Report Agreement Checks

To programmatically verify that all Markdown reports strictly match the underlying JSON outputs:
```bash
py -3 -m pytest vlm/tests/test_report_consistency.py -v
```
*Expected Result:* `2 passed in ~0.5s`.
