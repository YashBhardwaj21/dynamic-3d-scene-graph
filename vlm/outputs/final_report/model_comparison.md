# Visual Movability Classification: Qwen2.5-VL-3B Evaluation

**Run Directory:** `paper_benchmark_20261019_144106`  
**Evaluator Ground Truth:** `paper_benchmark_ground_truth_expanded.json`  
**Total Unique Physical Instances:** `42`

---

## Executive Summary

This report evaluates **Qwen2.5-VL-3B-Instruct** across two primary multimodal view configurations:
1. `CROP_ONLY`: Tightly bounded bounding-box crop of the target object.
2. `CROP_PLUS_CONTEXT`: Object crop paired with wide-angle contextual scene view.

---

### 1. Overall Performance Across View Modes

| Model | View Mode | Overall Accuracy [95% CI] | Balanced Accuracy | Macro-F1 | Median Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Qwen2.5-VL-3B** | `CROP_ONLY` | **95.2%** `[84.2%, 98.7%]` | **98.2%** | **90.7%** | 23409.2 |
| **Qwen2.5-VL-3B** | `CROP_PLUS_CONTEXT` | **95.2%** `[84.2%, 98.7%]` | **98.2%** | **90.7%** | 18509.6 |

---

### 2. Class-Wise Precision, Recall, and F1 Breakdown

| Model | View Mode | IMMOVABLE (P/R/F1) | MOVABLE (P/R/F1) | MOBILE (P/R/F1) |
| :--- | :--- | :---: | :---: | :---: |
| Qwen2.5-VL-3B | `CROP_ONLY` | 60.0% / 100.0% / 75.0% | 100.0% / 94.6% / 97.2% | 100.0% / 100.0% / 100.0% |
| Qwen2.5-VL-3B | `CROP_PLUS_CONTEXT` | 60.0% / 100.0% / 75.0% | 100.0% / 94.6% / 97.2% | 100.0% / 100.0% / 100.0% |

---

### 3. Operational Reliability & Wire Quality

| Model | View Mode | Requests Attempted | Valid Schema (%) | Accepted Predictions | Abstentions |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B | `CROP_ONLY` | 42 | 42/42 (100.0%) | 42 | 0 (0.0%) |
| Qwen2.5-VL-3B | `CROP_PLUS_CONTEXT` | 42 | 42/42 (100.0%) | 42 | 0 (0.0%) |

---

### 4. End-to-End Client Latency Breakdown

| Model | View Mode | Mean Latency (ms) | Median Latency (ms) | p95 Latency (ms) | Serving Backend |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Qwen2.5-VL-3B | `CROP_ONLY` | 20010.4 | 23409.2 | 26992.1 | llama.cpp (WSL Linux) |
| Qwen2.5-VL-3B | `CROP_PLUS_CONTEXT` | 18678.9 | 18509.6 | 19921.5 | llama.cpp (WSL Linux) |

---

### 5. Published Visual Classification Figures

The following figures have been generated in `figures/`:
1. `grouped_metrics_comparison.png`: Overall Accuracy, Balanced Accuracy, and Macro-F1 across view modes.
2. `confusion_matrices_crop_plus_context.png`: Normalized confusion matrices.
3. `per_class_recall_comparison.png`: Class-wise recall across IMMOVABLE, MOVABLE, and MOBILE.
4. `latency_percentiles_comparison.png`: End-to-end inference latency percentiles (mean, median, p95).
5. `accuracy_vs_latency_tradeoff.png`: Macro-F1 versus inference latency trade-off curve.

---
*Report automatically compiled from immutable wire inference records.*
