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

## 2. Paper-Grade Benchmark Protocols & Execution

The primary evaluation suite is executed through the unified orchestrator `vlm/scripts/run_paper_benchmark.py`:

### Protocol 1: Live Multimodal Visual Movability Classification (`visual-live`)
Executes strict live inference across evaluated vision models (Qwen2.5-VL-3B-Instruct on llama.cpp, Gemma 3 4B on Ollama) over the wire with dual hashing (source asset and wire payload SHA-256):
```bash
python vlm/scripts/run_paper_benchmark.py --protocol visual-live --models qwen2_5_vl_3b gemma3_4b --view-modes CROP_ONLY CROP_PLUS_CONTEXT
```
*Outputs:* `visual_classification_predictions_<model>_<view_mode>.jsonl`, `visual_classification_summary.json`, `run_manifest.json`.

---

### Protocol 2: Deterministic Policy Evaluation Replay (`policy-replay`)
Evaluates estimator-aware router policies and downstream SLAM landmark lifecycles from immutable recorded visual predictions. Strictly enforces copying of source prediction files into the run directory for standalone reproducibility:
```bash
python vlm/scripts/run_paper_benchmark.py --protocol policy-replay --replay-source-dir vlm/outputs/benchmark_runs/<prior_visual_run>
```
*Outputs:* `router_policies_summary.json`, `downstream_lifecycle_summary.json`, `statistical_metrics_summary.json`, `claim_to_evidence_matrix.md`, `paper_benchmark_report.md`, `run_manifest.json`.

---

### Protocol 3: Dynamic Live Router Evaluation (`router-live`)
Executes selective routing with dynamic dispatch, issuing live multimodal VLM queries strictly when the policy decides to escalate:
```bash
python vlm/scripts/run_paper_benchmark.py --protocol router-live --models qwen2_5_vl_3b
```

---

## 3. Historical Development & Diagnostic Scripts

The following scripts represent early development diagnostics and historical exploratory baselines:
- `vlm/scripts/run_stage6_ablation.py`: Initial Stage 6 development ablations.
- `vlm/scripts/run_vlm_controlled_ablations.py`: Development prompt formulation ablations.
- `vlm/scripts/evaluate_downstream_slam_effects.py`: Diagnostic downstream simulation tests.
- `vlm/scripts/run_full_experiment.py`: Multi-session pipeline orchestrator.

---

## 4. Verification & Full Test Suite

To execute the complete 217-test verification suite covering zero-leakage, cryptographic provenance, inference wire quality, router gating, empirical SLAM lifecycle, statistical bootstrapping, and report consistency:
```bash
python -m pytest vlm/tests/ -v
```
*Expected Result:* `217 passed in ~60s (100% passing)`.
