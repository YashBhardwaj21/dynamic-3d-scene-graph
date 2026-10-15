# Issue Register: VLM Subsystem & Experimental Pipeline

This register tracks all confirmed and suspected defects, their technical root causes, verified evidence, proposed minimal corrections, and regression test coverage.

---

## Issue Summary Matrix

| Issue ID | Severity | Component / Source File | Short Title | Status |
| :--- | :--- | :--- | :--- | :--- |
| `ISSUE-01` | `CRITICAL` | `vlm/src/cross_session_association.py` | Low Association Recall (23.33%) & High False Split Rate (76.67%) | `CONFIRMED (UPSTREAM)` |
| `ISSUE-02` | `HIGH` | `vlm/src/selective_router.py` | Visual Input Hard-Capped at 2 Views & Lacking Decoding Validation | `RESOLVED (PHASE 2)` |
| `ISSUE-03` | `HIGH` | `vlm/src/semantic_evidence.py` | Misleading 99.76% Macro-F1 Masking 44.43% Out-of-Vocabulary Records | `RESOLVED (PHASE 3)` |
| `ISSUE-04` | `HIGH` | `vlm/src/router_policy.py` | Missing Estimator-Aware Query Utility Formulation | `RESOLVED (PHASE 4)` |
| `ISSUE-05` | `HIGH` | `vlm/src/cross_session_association.py` | Boundary Trajectory SLAM Drift Corrupting SE(2) Registration (0.618m) | `CONFIRMED (UPSTREAM)` |
| `ISSUE-06` | `MEDIUM` | `vlm/src/contextual_scene_graph_fusion.py` | 30.42% Mean Relocation Displacement Error Driven by 2D Bounding Jitter | `CONFIRMED (UPSTREAM)` |
| `ISSUE-07` | `MEDIUM` | `vlm/scripts/evaluate_multiview_affordance.py` | Single-Class Bias in 28-Instance Held-Out Set (100% MOVABLE Furniture) | `RESOLVED (PHASE 5)` |
| `ISSUE-08` | `MEDIUM` | `vlm/src/temporal_evidence.py` | High Front-End Tracker Fragmentation (72.78% tracks $\le 4$ frames) | `CONFIRMED (UPSTREAM)` |
| `ISSUE-09` | `MEDIUM` | `vlm/src/selective_router.py` | VLM Cache Key Lacking Image Content Hashes & Model Artifact Metadata | `RESOLVED (PHASE 2)` |
| `ISSUE-10` | `LOW` | `vlm/src/crop_harvester.py` | Foreground Obstacle Clutter in Multi-View Crops | `DOCUMENTED (SUSPECTED)` |

---

## Detailed Issue Records

### ISSUE-01: Low Cross-Session Association Recall (23.33%) & High False Split Rate (76.67%)
- **Severity:** `CRITICAL`
- **Source File & Function:** `vlm/src/cross_session_association.py` (`associate_cross_session_candidates`, `evaluate_cross_session_association`)
- **Observed Behavior:** Across all 4 data association baselines (Centroid, Hungarian, Uncertainty Symmetric, Relocation Aware), only 7 out of 30 multi-session physical instances are successfully matched across sessions. 17 instances are dropped by spatial gates.
- **Evidence:** `vlm/outputs/full_experiment_run/per_instance_association_diagnostics.json` records 17 `UNMATCHED_GATE` events and 6 `FALSE_MERGE` events. `association_baselines_summary.json` confirms `all_instance_recall_pct = 23.33%`, `false_split_rate_pct = 76.67%`.
- **Root Cause:** Upstream tracking produces short, fragmented track bursts (72.78% $\le 4$ frames). Track centroids have high variance and fail to satisfy the rigid 0.45m spatial gate.
- **Minimal Justified Correction:** Implement temporal cluster aggregation prior to inter-session matching to consolidate fragmented tracklets of the same physical entity into a unified spatial centroid before gating.
- **Regression Test:** `vlm/tests/test_cross_session_association.py::test_cross_session_association_instance_recall`
- **Verification Command:** `pytest vlm/tests/test_cross_session_association.py -v`
- **Documentation Updated:** `vlm/docs/ENGINEERING_AUDIT.md`, `vlm/assignment/docs/KNOWN_ISSUES.md`
- **Remaining Limitations:** Does not resolve upstream camera motion blur during fast yaw turns ($> 0.5$ rad/s).

---

### ISSUE-02: Visual Input Hard-Capped at 2 Views & Lacking Decoding Validation
- **Severity:** `HIGH`
- **Source File & Function:** `vlm/src/selective_router.py` (`prepare_multiview_input`, `route_instance`)
- **Observed Behavior:** The router selects at most two consecutive view pairs (`crop_view_0` + `scene_view_0`, `crop_view_1` + `scene_view_1`). If an instance has 3 valid views, the 3rd view is silently dropped. If an image file is corrupt or empty, it is not rejected prior to inference.
- **Evidence:** `selective_router.py` hardcodes `max_views = 2` in visual assembly; missing image paths can produce an empty image list that is sent as text-only prompt while being counted as visual inference.
- **Root Cause:** Initial implementation hardcoded 2-view concatenation to fit context window limits without dynamic view count selection or input validation.
- **Minimal Justified Correction:** Add explicit view-count policy parameter (`view_count in [1, 2, 3]`), verify image file existence and image decoding via PIL/OpenCV, compute SHA-256 hashes of selected views, and reject inference if required views fail to decode.
- **Regression Test:** `vlm/tests/test_selective_router.py::test_view_selection_and_decoding_validation`
- **Verification Command:** `pytest vlm/tests/test_selective_router.py -v`
- **Documentation Updated:** `vlm/docs/PIPELINE_AND_DATA_FLOW.md`
- **Remaining Limitations:** Dependent on available harvested views in `multiview_samples/`.

---

### ISSUE-03: Misleading 99.76% Macro-F1 Masking 44.43% Out-of-Vocabulary Records
- **Severity:** `HIGH`
- **Source File & Function:** `vlm/src/semantic_evidence.py` (`compute_semantic_evidence`), `vlm/scripts/evaluate_stage5_evidence.py`
- **Observed Behavior:** Reports claim 99.76% semantic Macro-F1, implying near-perfect semantic understanding.
- **Evidence:** `vlm/outputs/stage5_evidence_metrics.json` shows `matched_observations_count = 4,421` out of `total_records = 7,956`. `unmatched_observations_count = 3,535` ($44.43\%$).
- **Root Cause:** The evaluator dropped all observations whose raw detector labels did not match the 3 evaluated taxonomy classes (`IMMOVABLE`, `MOVABLE`, `MOBILE`), computing Macro-F1 strictly over the in-vocabulary subset ($55.57\%$ coverage).
- **Minimal Justified Correction:** Update reporting contract to explicitly mandate dual reporting: conditional Macro-F1 on matched support alongside total record count, unmapped count, coverage percentage, and unknown rate.
- **Regression Test:** `vlm/tests/test_report_consistency.py::test_semantic_coverage_and_support_reporting`
- **Verification Command:** `pytest vlm/tests/test_report_consistency.py -v`
- **Documentation Updated:** `vlm/docs/METRIC_DICTIONARY.md`, `vlm/assignment/docs/METRIC_DICTIONARY.md`
- **Remaining Limitations:** Detector taxonomy still omits miscellaneous object categories (e.g., small desk clutter).

---

### ISSUE-04: Missing Estimator-Aware Query Utility Formulation
- **Severity:** `HIGH`
- **Source File & Function:** `vlm/src/router_policy.py` (`RouterPolicy.evaluate_decision`)
- **Observed Behavior:** The existing router uses rule-based confidence thresholds and heuristic priors. It has no signal indicating whether an observation actually affects SLAM state estimation or trajectory loss.
- **Evidence:** Inspection of `RouterPolicy` reveals decisions are based purely on $P(movable)$, semantic entropy, and spatial stability, without any parameter representing landmark Hessian information, Fisher information, or residual error.
- **Root Cause:** The selective router was developed as a movability classifier rather than an estimator-aware perception filter.
- **Minimal Justified Correction:** Formulate and implement explicit estimator-aware routing criterion:
  $$\text{Query Utility} = \Delta \mathcal{L}_{\text{SLAM}} - \lambda \cdot C_{\text{VLM}}$$
  where $\Delta \mathcal{L}_{\text{SLAM}}$ is proxied by landmark parallax, observation count, and spatial residual uncertainty.
- **Regression Test:** `vlm/tests/test_router_policy.py::test_estimator_aware_utility_dependency`
- **Verification Command:** `pytest vlm/tests/test_router_policy.py -v`
- **Documentation Updated:** `vlm/docs/PROJECT_OVERVIEW.md`, `vlm/docs/RESEARCH_RESULTS.md`
- **Remaining Limitations:** Offline proxy requires counterfactual validation without full real-time SLAM relocalization loop.

---

### ISSUE-05: Boundary Trajectory SLAM Drift Corrupting SE(2) Registration (0.618m)
- **Severity:** `HIGH`
- **Source File & Function:** `vlm/src/cross_session_association.py` (`compute_robust_se2`, `compute_anchor_residuals`)
- **Observed Behavior:** Anchors near trajectory boundaries (e.g., `desk_023` $\leftrightarrow$ `desk_034`) exhibit registration residuals up to 0.618m, exceeding the 0.45m static association gate.
- **Evidence:** `vlm/outputs/stage5_registration_audit.json` records 3 boundary outliers (residuals: 0.3897m, 0.5643m, 0.6180m), pulling all-anchor RMS to 0.3087m despite consensus inlier RMS being 0.1198m.
- **Root Cause:** Visual odometry drift accumulates at the physical periphery of the room where visual feature overlap is sparse.
- **Minimal Justified Correction:** Enforce RANSAC consensus SVD refitting strictly on consensus inliers for the global transform, and introduce anisotropic covariance-aware spatial gates that expand along the direction of drift.
- **Regression Test:** `vlm/tests/test_cross_session_association.py::test_registration_stability`
- **Verification Command:** `pytest vlm/tests/test_cross_session_association.py -v`
- **Documentation Updated:** `vlm/docs/ENGINEERING_AUDIT.md`
- **Remaining Limitations:** Does not rewrite upstream RTAB-Map bundle adjustment.

---

### ISSUE-06: 30.42% Mean Relocation Displacement Error Driven by 2D Bounding Jitter
- **Severity:** `MEDIUM`
- **Source File & Function:** `vlm/src/contextual_scene_graph_fusion.py` (`_calculate_displacement`)
- **Observed Behavior:** Estimated relocation displacements deviate from physical tape ground truth by 0.054m to 0.196m (mean relative error 30.42%).
- **Evidence:** Controlled events in `vlm/outputs/stage5_evidence_metrics.json` show Chair 002 (9.52% error), Chair 003 (32.34% error), Box 007 (49.42% error).
- **Root Cause:** Centroid estimation uses 2D detector bounding box midpoints projected to 3D without 3D point cloud segmentation or outlier point removal.
- **Minimal Justified Correction:** Implement statistical median outlier filtering on projected point clouds and apply bounding box temporal smoothing across observation sequences.
- **Regression Test:** `vlm/tests/test_scene_graph_integration.py::test_controlled_displacement_error_bounds`
- **Verification Command:** `pytest vlm/tests/test_scene_graph_integration.py -v`
- **Documentation Updated:** `vlm/docs/METRIC_DICTIONARY.md`
- **Remaining Limitations:** Sub-centimeter precision requires high-density LiDAR or dense mesh reconstruction.

---

### ISSUE-07: Single-Class Bias in 28-Instance Held-Out Set (100% MOVABLE Furniture)
- **Severity:** `MEDIUM`
- **Source File & Function:** `vlm/scripts/evaluate_multiview_affordance.py`
- **Observed Behavior:** The 28 held-out instances achieve 100% Balanced Accuracy and Macro-F1, but all 28 instances belong to a single class (`MOVABLE`).
- **Evidence:** `vlm/outputs/multiview_affordance_results.json` records 28 `MOVABLE`, 0 `IMMOVABLE`, 0 `MOBILE`.
- **Root Cause:** The held-out split was populated by taking all non-manifest furniture objects without stratification across the 3 ontology classes.
- **Minimal Justified Correction:** Create an authoritative stratified benchmark split containing verified representations of `IMMOVABLE`, `MOVABLE`, and `MOBILE` instances, preventing single-class trivial accuracy.
- **Regression Test:** `vlm/tests/test_benchmark_dataset.py::test_multiclass_split_stratification`
- **Verification Command:** `pytest vlm/tests/test_benchmark_dataset.py -v`
- **Documentation Updated:** `vlm/docs/DATA_AND_EVALUATION_INVENTORY.md`
- **Remaining Limitations:** Dataset has limited architectural `IMMOVABLE` instances (doors, walls, ceiling fans) due to indoor recording constraints.

---

### ISSUE-08: High Front-End Tracker Fragmentation (72.78% tracks $\le 4$ frames)
- **Severity:** `MEDIUM`
- **Source File & Function:** `vlm/src/temporal_evidence.py` (`compute_temporal_evidence`)
- **Observed Behavior:** 1,008 out of 1,385 tracks survive 4 or fewer frames before re-initialization under a new track ID.
- **Evidence:** `vlm/outputs/stage5_evidence_metrics.json` records `fragmented_tracks_count = 1008`, `track_fragmentation_rate_pct = 72.78%`.
- **Root Cause:** Visual occlusion and rapid camera panning break 2D IoU associations in the front-end SORT/ByteTrack tracker.
- **Minimal Justified Correction:** Add temporal persistence smoothing and inter-tracklet association within single sessions.
- **Regression Test:** `vlm/tests/test_stage5_evidence.py::test_temporal_persistence_and_fragmentation`
- **Verification Command:** `pytest vlm/tests/test_stage5_evidence.py -v`
- **Documentation Updated:** `vlm/docs/ENGINEERING_AUDIT.md`
- **Remaining Limitations:** Requires upstream detector tracker tuning for live operation.

---

### ISSUE-09: VLM Cache Key Lacking Image Content Hashes & Model Artifact Metadata
- **Severity:** `MEDIUM`
- **Source File & Function:** `vlm/src/selective_router.py` (`_compute_cache_key`)
- **Observed Behavior:** Cache keys are formed from string instance IDs and prompt text strings without hashing actual image binary bytes or model checkpoint version.
- **Evidence:** `router_vlm_cache.json` uses entity IDs as keys; modifying an underlying crop image does not invalidate the cache.
- **Root Cause:** Simplistic cache lookup designed for static file paths.
- **Minimal Justified Correction:** Include SHA-256 hashes of all input images, the prompt version string, and the model artifact identifier in the cache key.
- **Regression Test:** `vlm/tests/test_selective_router.py::test_cache_key_invalidation_on_image_change`
- **Verification Command:** `pytest vlm/tests/test_selective_router.py -v`
- **Documentation Updated:** `vlm/docs/PIPELINE_AND_DATA_FLOW.md`
- **Remaining Limitations:** Replay mode must verify hashes before reusing cached responses.

---

### ISSUE-10: Foreground Obstacle Clutter in Multi-View Crops
- **Severity:** `LOW (SUSPECTED)`
- **Source File & Function:** `vlm/src/crop_harvester.py` (`harvest_multiview_crops`)
- **Observed Behavior:** Certain crops centered on chairs include foreground desk edges or cables, confusing unguided VLM reasoning.
- **Evidence:** Multi-view crops in `vlm/outputs/multiview_samples/` exhibit partial occlusions by static desks.
- **Root Cause:** 2D bounding box cropping captures all pixels in the rectangular window without instance segmentation alpha masking.
- **Minimal Justified Correction:** Implement depth-guided foreground masking or prompt the VLM with an explicit bounding box visual overlay.
- **Regression Test:** `vlm/tests/test_multiview_crops.py::test_crop_visual_integrity`
- **Verification Command:** `pytest vlm/tests/test_multiview_crops.py -v`
- **Documentation Updated:** `vlm/docs/KNOWN_ISSUES.md`
- **Remaining Limitations:** Masking quality depends on YOLO segmentation mask accuracy.
