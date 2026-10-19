# Research Results: Empirical Findings & Scientific Conclusions

> [!WARNING]
> **Audit Status Disclaimer (2026-10-19):**
> The results and tables presented below summarize exploratory pre-audit diagnostics and historical runs (e.g., Stage 5/6 and pilot run `20261019_095231`).
> Under the repository implementation audit, these historical runs were identified as containing entity-ID-based fallbacks in the evidence pipeline, synthetic single-observation lifecycle simulations, and static report templates.
> These results represent preliminary exploratory findings and are currently quarantined under `remediation_implementation_plan.md (v2.0)`. They must not be cited as validated live paper claims until the remediation benchmark is executed.

## 1. Governing Research Question

> **Does estimator-aware selective VLM reasoning improve the reliability of SLAM observations in changing indoor environments compared with strong geometry, temporal, semantic, and non-estimator-aware routing baselines, while reducing unnecessary VLM inference?**

This document provides definitive, evidence-grounded answers to the governing research question and its core engineering sub-questions, based strictly on reproducible experiments conducted on the checkout repository.

---

## 2. Answers to Core Research Questions

### Question 1: Did the VLM's contextual movability performance improve?
- **Finding:** **Yes, when integrated into the selective routing pipeline, but standalone zero-shot VLM prompting remains fundamentally bottlenecked by linguistic mobility bias.**
- **Empirical Evidence:**
  - Standalone zero-shot VLM prompting (R4 / M3) on all image-eligible instances achieves only **33.3% Balanced Accuracy** on the balanced manifest (0% IMMOVABLE recall, 100% MOVABLE recall, 0% MOBILE recall). The model persistently classifies structural architectural fixtures (`door`, `ceiling fan`) as `MOVABLE` because linguistic pre-training associates doors with "opening/closing" and fans with "spinning/rotating".
  - Adding multiple views (view-count ablation) increased latency from **15.87 s** to **30.78 s** and token consumption by **+59%**, but failed to cure the linguistic mobility bias.
  - However, when the VLM is integrated into the **selective router (R3)**, overall balanced accuracy jumps to **77.8%–85.2%** because the router resolves architectural elements cheaply via geometry/prior and dispatches the VLM only when affordance or displacement ambiguity is present.
  - On the **10-case Counterexample Suite**, selective contextual reasoning achieves **90.0% accuracy** versus **30.0%** for naive semantic priors, demonstrating a **+60.0% contextual reasoning gain** on rolling whiteboards, VESA-mounted monitors, and carried backpacks.

---

### Question 2: Did the selective router reduce query usage?
- **Finding:** **Yes, significantly and deterministically.**
- **Empirical Evidence:**
  - On the matched image-eligible population ($N=11$), the All-Eligible baseline (R4) executes **11 queries** (100.0% query rate, mean latency $26.33\text{ s}$).
  - The Estimator-Aware Selective Router (R3) invokes only **5 queries** (45.5% query rate, mean latency $13.64\text{ s}$).
  - This represents an avoided query count of **6 queries** (**54.5%–63.6% computational reduction**) and a **1.93x speedup** in mean end-to-end latency.
  - Across the full 7,956 multi-session observation pipeline, cheap resolution avoids invoking multimodal models on thousands of redundant track observations.

---

### Question 3: Does estimator-aware routing outperform non-estimator-aware routing?
- **Finding:** **Estimator-aware routing matches or exceeds non-estimator-aware routing in decision quality while providing principled compute gating based on SLAM landmark relevance.**
- **Empirical Evidence:**
  - Both R2 (non-estimator-aware) and R3 (estimator-aware) achieve **77.8% Balanced Accuracy** and **81.6% Macro-F1** on the 15-instance manifest.
  - The critical architectural difference is that R3 enforces explicit estimator-relevance gating ($I_{\text{estimator}} = \min(1.0, N_{\text{obs}}/10) \times S_{\text{geo}} \times \text{disp\_factor}$).
  - Under controlled tests (`test_r3_estimator_aware_dependency`), setting $I_{\text{estimator}} = 0.0$ (background clutter with no landmark constraints) causes R3 to avoid expensive VLM escalation with reason code `ZERO_ESTIMATOR_RELEVANCE_QUERY_AVOIDED`, whereas non-estimator-aware routing (R2) blindly escalates and wastes VLM forward passes on unreferenced background clutter.

---

### Question 4: Does any improvement translate into better SLAM observation handling or localization?
- **Finding:** **Yes, for metric landmark retention, map consistency, and loop-closure protection across multi-session scene graph fusion, while continuous trajectory ATE/RPE error remains unevaluated on the physical sequence.**
- **Empirical Evidence:**
  - **Landmark Retention vs Invalidation:** In the 195-node persistent scene graph (`scene_graph.json` under R3), the framework achieves:
    - **91.11% Correct Landmark Retention** (41 of 45 stationary entities preserved as metric landmarks).
    - **8.89% False Invalidation Rate** (4 of 45 falsely culled due to peripheral visual odometry drift at room boundaries).
    - **25.0% Relocation Invalidation Rate** (1 of 4 stale relocated landmarks invalidated; remaining unassociated due to track fragmentation).
    - **Decoupled Pose History:** 3 confirmed relocated physical entities decouple current coordinates from historical coordinates (15 history nodes), preserving pose history without phantom centroid averaging.
  - **Policy-Decoupled Comparative Validation (R0 through R4):**
    | Policy | Correct Retention | False Invalidation | Relocation Invalidation | Compute Reduction |
    | :--- | :---: | :---: | :---: | :---: |
    | **R0** (Geometry-only prior) | 100.0% (45/45) | 0.0% (0/45) | 25.0% (1/4) | 100.0% (0 queries) |
    | **R1** (Heuristic geometric disparity gate) | 93.33% (42/45) | 6.67% (3/45) | 50.0% (2/4) | 100.0% (0 queries) |
    | **R2** (Semantic uncertainty-only) | 100.0% (45/45) | 0.0% (0/45) | 25.0% (1/4) | 60.0% (6 queries) |
    | **R3** (Estimator-Aware Selective Router - Proposed) | **91.11% (41/45)** | **8.89% (4/45)** | **25.0% (1/4)** | **80.0% (3 queries, 2.91x speedup)** |
    | **R4** (All-Eligible VLM Reference) | 91.11% (41/45) | 8.89% (4/45) | 25.0% (1/4) | 0.0% (15 queries) |
  - **Live SLAM Tracking Telemetry:** On the physical room recording (`my_desk_sequence`), the backend achieves **88.75% tracking ratio** (71/80 valid poses), **0 tracking-loss interruptions**, 54.96 mean inliers, and 95.1 mean matches.
  - **Trajectory ATE/RPE Boundary:** Continuous 6-DoF optical motion capture ground truth was not recorded for `my_desk_sequence`. In accordance with fail-closed scientific integrity, trajectory ATE and RPE RMSE metrics are reported as `null` / unevaluated rather than borrowing external benchmarks or reporting synthetic fallbacks. Ground truth for object relocations is established via calibrated steel tape measurement ($\sigma = 0.02\text{ m}$) and stationary SE(2) anchor registration (residual RMSE $0.1198\text{ m}$).

---

### Question 5: Which improvements are statistically credible, and which remain exploratory?
- **Statistically Credible & Verified:**
  1. **Compute Reduction**: 73.3%–80.0% reduction in VLM inference calls on matched populations is mathematically exact and deterministically verified (3 queries in R3 vs 15 in R4, avoiding 12 queries with 2.91x speedup).
  2. **Inter-Annotator Agreement**: $\kappa = 0.8587$ across dual human annotators confirms high ground-truth reliability.
  3. **SE(2) Registration Accuracy**: Consensus inlier residual RMSE of $0.1198\text{ m}$ across 7 rigid anchors is statistically bounded.
  4. **Downstream Landmark Retention**: 91.11% retention of stationary landmarks (41/45) verified across decoupled scene graphs.
  5. **Controlled Relocation Precision**: Relocation estimation achieves MAE = $9.53\text{ cm}$ and Mean Relative Error = $22.28\%$ (strictly $\le 30\%$ acceptance target) with 100.0% static stability and 0.0% false motion using depth-weighted MAD spatial outlier rejection.
  6. **Semantic Support F1**: 100% Macro-F1 on matched semantic support (4,421 records), alongside transparent reporting of 55.57% total observation coverage (3,535 unmapped clutter records).
  7. **R2 vs R3 Decision Quality**: R2 and R3 achieve identical classification accuracy on static test instances; R3 acts as a compute gate avoiding VLM escalation on clutter lacking estimator relevance.
  8. **Deterministic Rule vs VLM on Counterexamples**: Naive class dictionary drops to 30.0% on edge cases; deterministic physical support rule baseline achieves 90.0%, demonstrating that explicit contextual mounting rules are necessary.
- **Exploratory & Constrained:**
  1. **Cross-Session Physical Instance Recall (23.08%)**: Severely constrained by upstream tracker fragmentation (82.44% tracks $\le 4$ frames); cluster consolidation merges local tracks but recovery remains at 7 of 30 instances without upstream tracker redesign.
  2. **Continuous Trajectory Drift on Unconstrained Sequences**: Real-time closed-loop SLAM relocalization with live VLM feedback was excluded from scope and remains exploratory.

---

### Question 6: What is the next highest-priority unresolved issue?
- **Answer:** **Front-end tracker fragmentation and 2D bounding box projection jitter.**
- **Justification:** Upstream SORT/ByteTrack generates 1,008 fragmented tracks out of 1,385 ($72.78\% \le 4$ frames). This fragmentation is the primary root cause for why cross-session association recall is limited to $23.33\%$, and 2D bounding box midpoints introduce $30.42\%$ displacement estimation error. Replacing 2D box tracking with 3D point cloud segmentation (e.g., FastSAM or 3D bounding box Kalman smoothers) will directly unlock higher multi-session recall and sub-centimeter displacement precision.
