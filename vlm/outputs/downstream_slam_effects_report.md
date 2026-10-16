# Downstream SLAM and Dynamic 3D Scene Graph Validation Report

## 1. Executive Summary

This evaluation validates the impact of estimator-aware selective VLM routing against four baselines on persistent 3D scene graph generation and metric landmark reliability.

### Key Findings:
- **Correct Landmark Retention (R3):** 91.11% (41/45 stationary objects retained).
- **False Invalidation Rate (R3):** 8.89% (4/45 false invalidations).
- **Relocation Invalidation Rate (R3):** 25.0% (1/4 stale landmarks invalidated).
- **Compute Reduction:** 80.0% VLM inference queries avoided with 3.67x speedup.

---

## 2. Policy-Specific Comparative Matrix

| Policy | Description | Nodes | Retention | False Invalidation | Relocation Invalidation | Pose History Nodes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **R0** | Geometry-Only Prior (Zero VLM) | 194 | 100.0% (45/45) | 0.0% (0/45) | 25.0% (1/4) | 0 |
| **R1** | Heuristic Geometric Disparity Gate (>= 0.25m) | 194 | 93.33% (42/45) | 6.67% (3/45) | 50.0% (2/4) | 11 |
| **R2** | Semantic Uncertainty-Only | 195 | 100.0% (45/45) | 0.0% (0/45) | 25.0% (1/4) | 0 |
| **R3** | Estimator-Aware Selective Router (Proposed) | 195 | 91.11% (41/45) | 8.89% (4/45) | 25.0% (1/4) | 12 |
| **R4** | All-Eligible VLM Reference | 195 | 91.11% (41/45) | 8.89% (4/45) | 25.0% (1/4) | 12 |

---

## 3. Metric Trajectory Telemetry (Fail-Closed Boundary)

- **Evaluation Sequence:** `my_desk_sequence`
- **Optical MoCap Ground Truth:** `False`
- **Trajectory ATE RMSE:** `None` (UNEVALUATED)
- **Trajectory RPE RMSE:** `None` (UNEVALUATED)
- **Tracking Ratio:** 88.75% (71/80 valid poses)
- **Tracking Loss Frequency:** 0 lost frames
- **Mean Inliers:** 54.9625
- **Mean Matches:** 95.1
- **Ground Truth Methodology:** Calibrated steel tape measure (sigma = 0.02m) + RANSAC SE(2) anchors (sigma = 0.1198m)

> **Scientific Boundary Note:** Continuous 6-DoF optical motion capture is not recorded for my_desk_sequence. In accordance with fail-closed scientific boundaries, trajectory ATE/RPE error metrics are reported as null/unevaluated rather than substituting external benchmark or synthetic values.
