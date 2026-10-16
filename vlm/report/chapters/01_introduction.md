# Chapter 1: Introduction

## 1.1 Motivation & Problem Statement
Visual Simultaneous Localization and Mapping (VSLAM) systems operating in human-inhabited indoor environments face a fundamental challenge: the physical world is non-static. Objects are routinely shifted, rearranged, carried, or removed across operational sessions. Conventional metric semantic SLAM pipelines routinely make binary assumptions regarding dynamics—either treating all non-rigid visual features as dynamic outliers to be masked out (such as moving humans or robots), or conversely assuming that all rigid spatial landmarks remain permanently stationary.

When furniture items (such as office chairs, mobile tables, and containers) are relocated between mapping sessions, conventional spatial fusion engines suffer from two catastrophic failure modes:
1. **False Landmark Invalidation**: Visual odometry drift and minor peripheral camera tracking jitter cause stationary structures to be falsely culled from the persistent map.
2. **Phantom Multi-Modal Averaging**: Displaced objects are fused across sessions into synthetic, non-physical spatial averages that corrupt downstream camera tracking and loop closure.

## 1.2 The Governing Research Question
This research investigates the following central scientific question:
> **Does estimator-aware selective Vision-Language Model (VLM) reasoning improve the reliability of SLAM observations in changing indoor environments compared with strong geometry, temporal, semantic, and non-estimator-aware routing baselines, while reducing unnecessary VLM inference?**

## 1.3 Hypotheses & Scope Boundaries
- **Hypothesis 1 (Contextual Mobility Disambiguation)**: Multi-modal contextual reasoning resolves fine-grained physical affordances (e.g., distinguishing wall-mounted fixtures from rolling whiteboards or carried luggage) that rigid semantic class dictionaries fail to separate.
- **Hypothesis 2 (Computational Query Gating)**: Estimator-influence gating avoids expensive multimodal inference forward passes on visual background clutter that provides zero constraint to the SLAM backend.
- **Hypothesis 3 (Metric Map Integrity)**: Uncertainty-adaptive multi-session association preserves verified stationary landmarks while selectively invalidating relocated landmarks to protect downstream map consistency.

### Explicit Scope Boundaries
1. **Continuous Trajectory Ground Truth**: Continuous 6-DoF optical motion capture ground truth was not recorded for the physical room sequence (`my_desk_sequence`). In strict adherence to fail-closed scientific boundaries, trajectory ATE and RPE RMSE metrics are reported as **UNEVALUATED** (`null`). Ground-truth metric evaluation is conducted directly on calibrated steel tape measurements ($\sigma = 0.02\text{ m}$) for object relocations and rigid SE(2) consensus anchor registration ($	ext{RMSE} = 0.1198\text{ m}$).
2. **Offline Replay vs. Live Edge Deployment**: Experiments in this package evaluate multimodal routing decisions over frozen observation sequences. Zero-shot VLM model latencies are explicitly documented as projected benchmark inference times; cached replays are never conflated with live measured wall-clock latencies.
3. **No Ground-Truth Leakage**: All policy comparisons (R0 through R4) operate strictly on sensor observations, spatial Kalman clusters, and registration uncertainty without querying benchmark annotations.

## 1.4 Principal Contributions
1. **Estimator-Aware Selective Routing Architecture (R3)**: A decision framework coupling predicted trajectory loss reduction $\Delta L_{\text{SLAM}}$ with inference cost $\lambda_{\text{cost}} C_{\text{VLM}}$ to selectively gate VLM dispatch.
2. **Registration-Uncertainty-Adaptive Multi-Session Association**: Dynamic spatial gating adapting to peripheral visual odometry drift, eliminating false merges on closely spaced furniture.
3. **Decoupled Persistent 3D Scene Graph Engine**: A lifecycle maintenance framework decoupling historical and current object poses to prevent phantom multi-modal centroid averaging.
4. **Exhaustive Empirical Benchmark Suite**: Controlled evaluations across 7,956 observations, 1,970 tracks, 52 annotated physical instances, and 10 contextual counterexamples.
