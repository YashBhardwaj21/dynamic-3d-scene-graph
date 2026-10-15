# Known Issues and Remaining Limitations

## 1. Overview

This document transparently records all known limitations, upstream defects, and unresolved bottlenecks in the Dynamic 3D Scene Graph system. Maintaining this record ensures that experimental results are not misconstrued and prevents unverified claims regarding capabilities that remain constrained by hardware or upstream tracking.

---

## 2. Unresolved Upstream Perception & Tracking Limitations

### 2.1 Front-End Tracker Fragmentation (72.78% Short Tracks)
- **Component:** Upstream SORT/ByteTrack 2D bounding box tracker (`vlm/outputs/object_observations.jsonl`).
- **Observed Behavior:** Out of 1,385 unique tracks across the two sessions, 1,008 tracks (72.78%) span 4 or fewer consecutive camera frames before terminating.
- **Downstream Consequence:** Spatial Kalman filters cannot accumulate enough consecutive depth observations to establish stable, low-variance 3D centroids. When tracks re-appear with new track IDs, cross-session association fails to link them to the true physical instance.
- **Scope Note:** Upstream tracking and real-time detector re-training are outside the offline VLM evaluation scope.

### 2.2 Low Cross-Session Association Recall (23.33% Physical Instance Recovery)
- **Component:** `vlm/src/cross_session_association.py`.
- **Observed Behavior:** Across all four matching baselines (Centroid Nearest Neighbor, Hungarian Assignment, Uncertainty-Symmetric Gating, and Relocation-Aware Matching), only 7 out of 30 multi-session physical instances are recovered across Session 1 and Session 2.
- **Downstream Consequence:** 23 physical instances remain split into separate session-specific entities, producing a high False Split Rate ($76.67\%$).
- **Evidence:** `vlm/outputs/full_experiment_run/association_baselines_summary.json` confirms:
  - Total multi-session instances: 30
  - Recovered instances: 7
  - Static instance recall: $23.08\%$ (6 / 26)
  - Relocated instance recall: $25.00\%$ (1 / 4)
  - Candidate precision: $58.33\%$ (7 / 12)

### 2.3 Boundary Trajectory Visual Odometry Drift in SE(2) Registration (0.618m)
- **Component:** Inter-session RANSAC SE(2) coordinate registration (`vlm/outputs/stage5_registration_audit.json`).
- **Observed Behavior:** While consensus inlier anchors achieve a low residual RMSE ($0.1198\text{ m}$ across 7 inliers), 3 boundary anchors located at the room's physical perimeter (`desk_023`, `desk_031`, `desk_034`) exhibit residuals between $0.3897\text{ m}$ and $0.6180\text{ m}$.
- **Root Cause:** RTAB-Map visual odometry accumulates rotational drift during sharp camera turns ($> 0.5\text{ rad/s}$) at trajectory boundaries where visual feature matches become sparse.
- **Mitigation Implemented:** RANSAC rigidly filters out the 3 boundary outliers during SVD transformation computation, protecting global alignment from peripheral drift.

### 2.4 Controlled Relocation Estimation Error (30.42% Mean Relative Error)
- **Component:** `vlm/src/contextual_scene_graph_fusion.py` (`_calculate_displacement`).
- **Observed Behavior:** For the 3 controlled physical object relocations, the estimated 3D centroid shifts deviate from physical tape-measured ground truth by:
  - Chair 002: Measured $0.6166\text{ m}$ vs Ground Truth $0.5630\text{ m}$ ($+0.0536\text{ m}$ error, $9.52\%$).
  - Chair 003: Measured $0.2612\text{ m}$ vs Ground Truth $0.3860\text{ m}$ ($-0.1248\text{ m}$ error, $32.34\%$).
  - Storage Box 007: Measured $0.5917\text{ m}$ vs Ground Truth $0.3960\text{ m}$ ($+0.1957\text{ m}$ error, $49.42\%$).
  - **Mean Absolute Error:** $0.1247\text{ m}$; **Mean Relative Error:** $30.42\%$.
- **Root Cause:** Centroid positions are derived from 2D YOLO bounding box midpoints projected to 3D depth, without dense 3D instance segmentation or point cloud outlier culling.

---

## 3. Dataset & Benchmark Coverage Limitations

### 3.1 Out-of-Vocabulary Detector Clutter (44.43% Unmapped Records)
- **Component:** `vlm/src/semantic_evidence.py` (`vlm/scripts/evaluate_stage5_evidence.py`).
- **Observed Behavior:** Out of 7,956 raw observation records, 3,535 records ($44.43\%$) have detector labels outside the 3 evaluated taxonomy classes (`IMMOVABLE`, `MOVABLE`, `MOBILE`).
- **Impact:** Headline macro-F1 of $99.76\%$ applies strictly to the matched support ($55.57\%$ coverage). Any claim that the system has $99.76\%$ whole-pipeline semantic accuracy is invalid without citing the coverage constraint.

### 3.2 Single-Class Composition in 28-Instance Held-Out Set
- **Component:** `vlm/scripts/evaluate_multiview_affordance.py`.
- **Observed Behavior:** The 28 held-out physical instances with crops (outside the 15-instance manifest) consist exclusively of `MOVABLE` furniture (15 chairs, 10 desks, 2 storage boxes, 1 backpack).
- **Impact:** 100% accuracy on this set is an artifact of single-class evaluation. Stratified evaluation on the 15-instance balanced manifest or the 10-case counterexample suite must be referenced to assess true discriminatory capability.

### 3.3 Lack of Optical Motion Capture on `my_desk_sequence`
- **Component:** Downstream SLAM evaluation.
- **Observed Behavior:** While optical MoCap trajectory ground truth exists for standard benchmark sequences (`tum_fr1_desk`), the physical room sequence (`my_desk_sequence`) relies on tape-measured ground truth for object relocations, not continuous 6-DoF optical tracking.
- **Boundary Enforced:** Trajectory ATE/RPE improvements are reported on the TUM benchmark, while map consistency and landmark invalidation are reported on the physical room sequence.

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
