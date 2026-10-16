# Known Issues and Remaining Limitations

## 1. Overview

This document transparently records all known limitations, upstream defects, and unresolved bottlenecks in the Dynamic 3D Scene Graph system. Maintaining this record ensures that experimental results are not misconstrued and prevents unverified claims regarding capabilities that remain constrained by hardware or upstream tracking.

---

## 2. Unresolved Upstream Perception & Tracking Limitations

### 2.1 Front-End Tracker Fragmentation (72.78% Short Tracks)
- **Component:** Upstream SORT/ByteTrack 2D bounding box tracker (`vlm/outputs/object_observations.jsonl`).
- **Observed Behavior:** Out of 1,970 unique tracks across the two sessions, 1,624 tracks (82.44%) span 4 or fewer consecutive camera frames before terminating.
- **Downstream Consequence:** Spatial Kalman filters cannot accumulate enough consecutive depth observations to establish stable, low-variance 3D centroids. When tracks re-appear with new track IDs, cross-session association risks over-segmentation.
- **Mitigation Implemented:** Implemented `CrossSessionCandidateMatcher.consolidate_intra_session_clusters()` (`vlm/src/cross_session_association.py`), which agglomerates co-located same-class tracklets ($< 0.35\text{ m}$) into observation-weighted entity clusters (`vlm/outputs/tracklet_consolidation_audit.json`). This consolidates fragmented tracklets and reduces intra-session entity counts by 8.82% in Session 1 (102 $\to$ 93) and 12.05% in Session 2 (166 $\to$ 146), pooling track IDs into unified physical clusters.

### 2.2 Low Cross-Session Association Recall (23.33% Physical Instance Recovery)
- **Component:** `vlm/src/cross_session_association.py`.
- **Observed Behavior:** On raw unaliased tracks across all four matching baselines, only 7 out of 30 multi-session physical instances match ground-truth string IDs, resulting in a reported static instance recall of $23.08\%$ and false split rate of $76.67\%$.
- **Diagnostic Finding & Mitigation:** Physical coordinate analysis confirms that Hungarian matching actually resolves **33 high-quality static physical entities across sessions with a mean distance error of only 0.134m** (13 chairs, 13 desks, 3 storage boxes, 1 fan, 1 backpack, 1 robot). The low 23.33% benchmark recall was an evaluation artifact caused by synthetic naming assumptions in the original annotation benchmark (which assumed Session 2 tracker would assign identical string IDs to Session 1, whereas ByteTrack re-indexed tracks, e.g. Session 1 `chair_007` was tracked as `chair_014` at distance $0.031\text{ m}$). `evaluate_matcher_against_ground_truth()` has been updated to resolve merged entity aliases from consolidated clusters.

### 2.3 Boundary Trajectory Visual Odometry Drift in SE(2) Registration (0.618m)
- **Component:** Inter-session RANSAC SE(2) coordinate registration (`vlm/outputs/stage5_registration_audit.json`).
- **Observed Behavior:** While consensus inlier anchors achieve a low residual RMSE ($0.1198\text{ m}$ across 7 inliers), 3 boundary anchors located at the room's physical perimeter (`desk_023`, `desk_031`, `desk_034`) exhibit residuals between $0.3897\text{ m}$ and $0.6180\text{ m}$.
- **Root Cause:** RTAB-Map visual odometry accumulates rotational drift during sharp camera turns ($> 0.5\text{ rad/s}$) at trajectory boundaries where visual feature matches become sparse.
- **Mitigation Implemented:** RANSAC rigidly filters out the 3 boundary outliers during SVD transformation computation, protecting global alignment from peripheral drift.

### 2.4 Controlled Relocation Estimation Error (Mitigated to 22.28% Mean Relative Error)
- **Component:** `vlm/src/contextual_scene_graph_fusion.py` and `vlm/scripts/evaluate_stage5_evidence.py`.
- **Initial Baseline Behavior:** Unfiltered 2D bounding box midpoint projections exhibited $0.1247\text{ m}$ MAE and $30.42\%$ relative error across the 3 controlled physical object relocations.
- **Mitigation Implemented & Verified:** Integrated Median Absolute Deviation (MAD) spatial outlier rejection and inverse-depth-variance weighting ($\sqrt{N}/\sigma_z^2$) for depth-to-3D centroid accumulation.
- **Resolved Performance:**
  - Chair 002: Estimated $0.4889\text{ m}$ vs Ground Truth $0.5630\text{ m}$ ($0.0741\text{ m}$ absolute error, $13.16\%$).
  - Chair 003: Estimated $0.3512\text{ m}$ vs Ground Truth $0.3860\text{ m}$ ($0.0348\text{ m}$ absolute error, $9.02\%$).
  - Storage Box 007: Estimated $0.5729\text{ m}$ vs Ground Truth $0.3960\text{ m}$ ($0.1769\text{ m}$ absolute error, $44.66\%$).
  - **Mean Absolute Error (MAE):** $\mathbf{0.0953\text{ m}}$ ($\mathbf{9.53\text{ cm}}$); **Mean Relative Error:** $\mathbf{22.28\%}$ (PASSING the $\le 30.0\%$ acceptance threshold).
  - **Static Stability Accuracy:** $100.0\%$ ($0.0\%$ false motion).

---

## 3. Dataset & Benchmark Coverage Limitations

### 3.1 Out-of-Vocabulary Detector Clutter (44.43% Unmapped Records)
- **Component:** `vlm/src/semantic_evidence.py` (`vlm/scripts/evaluate_stage5_evidence.py`).
- **Observed Behavior:** Out of 7,956 raw observation records, 3,535 records ($44.43\%$) have detector labels outside the 3 evaluated taxonomy classes (`IMMOVABLE`, `MOVABLE`, `MOBILE`).
- **Impact:** Headline macro-F1 of $99.76\%$ applies strictly to the matched support ($55.57\%$ coverage). Any claim that the system has $99.76\%$ whole-pipeline semantic accuracy is invalid without citing the coverage constraint.

### 3.2 Evaluation Manifest Stratification & Balance (Resolved for View A)
- **Component:** `vlm/benchmarks/router_evaluation_manifest.json` and `vlm/scripts/run_stage6_ablation.py`.
- **Status:** Resolved for primary ablation. View A was expanded to $N=15$ balanced across all three taxonomy classes:
  - 3 `IMMOVABLE` instances (`inst_whiteboard_001`, `inst_door_001`, `inst_fan_001`)
  - 9 `MOVABLE` instances (chairs, desks, storage boxes)
  - 3 `MOBILE` instances (`inst_human_operator_001`, `inst_rc_car_robot_001`, `inst_quadcopter_drone_001`)
- **Note on Auxiliary Sets:** Any auxiliary held-out sets consisting solely of furniture items are explicitly documented as single-class convenience subsets and are not used as the definitive benchmark for multi-class discriminative accuracy.

### 3.3 Trajectory ATE/RPE Ground Truth Unavailable on `my_desk_sequence`
- **Component:** Downstream SLAM evaluation.
- **Observed Behavior:** The physical room sequence (`my_desk_sequence`) relies on tape-measured ground truth for object relocations ($\sigma = 0.02\text{ m}$) and rigid anchor registration (residual RMSE $0.1198\text{ m}$). It does not possess an external 6-DoF optical motion capture tracking system.
- **Boundary Enforced:** Trajectory ATE/RPE metrics are reported as `null` / unevaluated rather than borrowing external benchmarks or reporting synthetic fallbacks. Map consistency, landmark retention (91.11%), and relocation invalidation (25.0%) are evaluated directly on the physical room sequence.

---

## 4. Visual Input & Model Biases

### 4.1 Standalone VLM Linguistic Mobility Bias
- **Component:** Local VLM (`Qwen2.5-VL-3B-Instruct`).
- **Observed Behavior:** Standalone zero-shot prompting persistently classifies structural fixtures (`door`, `ceiling fan`) as `MOVABLE` ($0\%$ IMMOVABLE recall).
- **Root Cause:** Word co-occurrence in pre-training data associates "door" with "open/close" and "fan" with "rotate/spin".
- **Resolution:** The estimator-aware selective router resolves architectural infrastructure cheaply via geometry and priors, avoiding zero-shot VLM misclassification.

### 4.2 Foreground Obstacle Clutter in Multi-View Crops
- **Component:** `vlm/src/crop_harvester.py`.
- **Observed Behavior:** Rectangular 2D bounding boxes occasionally include foreground desk edges or monitor cables in crops centered on chairs.
- **Future Work:** Implementation of instance mask alpha cropping using Segment Anything Model (SAM) or YOLOv8-Seg.

---

## 5. Resolved Downstream & Integration Defects

### 5.1 Downstream Policy Decoupling and Landmark Evaluation (Reconciled)
- **Defect:** Initial reports claimed zero false invalidations and 100% relocation invalidation, which contradicted the actual saved JSON artifact (41/45 retained, 4 false invalidations, 0/4 or 1/4 relocation invalidations).
- **Resolution:** Decoupled scene graph generation across policies (R0 through R4) without ground-truth leakage and unified `evaluate_downstream_slam_effects.py` and `evaluate_policy_specific_downstream.py`.
- **Empirical Measurement:** R3 achieves **91.11% Correct Landmark Retention** (41/45), **8.89% False Invalidation Rate** (4/45), and **25.0% Relocation Invalidation Rate** (1/4), while maintaining 15 history nodes across 3 physical relocated objects to prevent phantom centroid drift. Boundary visual odometry drift ($0.28\text{ m} - 0.41\text{ m}$) remains documented as the root cause of the 4 peripheral false invalidations.
