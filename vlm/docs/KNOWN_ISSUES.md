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

### 3.3 Trajectory ATE/RPE Ground Truth Unavailable on `my_desk_sequence`
- **Component:** Downstream SLAM evaluation.
- **Observed Behavior:** The physical room sequence (`my_desk_sequence`) relies on tape-measured ground truth for object relocations ($\sigma = 0.02\text{ m}$) and rigid anchor registration (residual RMSE $0.1198\text{ m}$). It does not possess an external 6-DoF optical motion capture tracking system.
- **Boundary Enforced:** Trajectory ATE/RPE metrics are reported as `null` / unevaluated rather than borrowing external benchmarks or reporting synthetic fallbacks. Map consistency, landmark retention (91.1%), and relocation invalidation (100.0%) are evaluated directly on the physical room sequence.

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
