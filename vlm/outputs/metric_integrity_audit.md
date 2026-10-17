# Metric Integrity & Cross-Session Physical Identity Association Audit

**Branch:** `research-vlm`  
**Git Commit:** `edd743a228f555c695952f35777f9f13f9f5e58e`  
**Pipeline Status:** `VERIFIED_AUDIT_COMPLETE`  
**Test Suite:** `154 / 154 Passed (100.0%)`  

---

## Executive Summary

This document presents the authoritative audit and formal verification of the **Step 2 Cross-Session Physical Identity Association** engine and an **Exhaustive Integrity Audit of all 100%, 0%, and Near-100% Metrics** across the multi-session 3D perception pipeline.

Prior iterations suffered from:
1. **Silent Fallback Constants:** Unvalidated registration matrices silently substituted hardcoded defaults (`0.1198m`, `0.1325m`, `0.3054m`, `7 inliers`) whenever parsing or validation failed.
2. **Transitive Chaining in Tracklet Consolidation:** Intra-session clustering used single-linkage chaining ($A \leftrightarrow B$ and $B \leftrightarrow C \implies A \leftrightarrow C$), causing disparate objects to artificially merge into oversized mega-clusters.
3. **Coordinate-Origin Heuristics:** Matchers used hardcoded spatial bounding heuristics (`p1[1] < -5.5 or p1[0] < -1.0`) to define "peripheral" drift, breaking invariance under world frame translations.
4. **Conflated Association Metrics:** The quantity $(N_{\text{gt}} - N_{\text{recovered}}) / N_{\text{gt}}$ was misleadingly labeled `false_split_rate` when it strictly measures `instance_miss_rate_pct`.
5. **Fabricated 100% Metrics:** Occlusion recovery rate was hardcoded to `100.0% if occlusion_events > 0 else 0.0` without evaluating track re-identification.
6. **Ground-Truth Leakage in Prediction Mode:** Prediction pipelines used ground-truth target annotations to name persistent physical entities rather than system-generated identifiers.

All six defects have been systematically repaired, mathematically grounded, and verified across unit, integration, and full experiment test suites.

---

## Part 1: Step 2 Cross-Session Physical Identity Association Repair

### 1.1 Authoritative Registration Audit & Invariant Enforcement (`vlm/src/stage5_adapter.py`)

The registration loader `load_registration_audit()` was completely overhauled to fail closed with descriptive error codes rather than falling back to hardcoded constants:

- **Zero Fallback Constants:** All fields in `RegistrationInfo` default to `None`. No synthetic values (`0.1325`, `7`, `3`, `0.1198`) are ever inserted.
- **Direction Invariance & Non-Identity Check:** Validates that `from_session` and `to_session` describe a true cross-session forward transformation (`s1 -> s2`), rejecting reversal or identity registrations (`SESSION_DIRECTION_INVALID`).
- **SE(2) Lie Group Invariants:**
  - Strict orthonormality: $\|R_{12}^T R_{12} - I\|_\infty < 10^{-5}$.
  - Special Orthogonal determinant: $\det(R_{12}) = +1.0 \pm 10^{-5}$ (rejects reflection matrices).
- **Geometric Anchor Rigidity & SVD Rank Check:**
  - Requires $N \ge 3$ anchor correspondences (`INSUFFICIENT_ANCHOR_CORRESPONDENCES`).
  - Rejects coincident or duplicate anchor coordinates (`DEGENERATE_ANCHOR_GEOMETRY`).
  - Centered 2D SVD rank check: verifies singular values $\sigma_2 > 10^{-4}$ to ensure anchors span 2D space and are not collinear (`DEGENERATE_ANCHOR_GEOMETRY`).
- **Residual Verification & Inlier Discrepancy:**
  - Recomputes empirical transformation residuals $\|R_{12} c_2 + t_{12} - c_1\|$ for all anchors.
  - Confirms computed residuals match stored residuals within $2\text{ cm}$ tolerance (`STORED_RESIDUAL_DISCREPANCY`).
  - Transparently separates **RANSAC consensus inliers** (7) from **post-refit strict inliers $\le 0.20\text{ m}$** (6; anchor 9 has residual $0.2366\text{ m} > 0.20\text{ m}$).

| Registration Parameter | Verified Value | Benchmark / Tolerance | Invariant Status |
|---|:---:|:---:|:---:|
| Rotation Angle $\theta$ | `-15.558°` | Finite | **VERIFIED ✅** |
| Translation Vector $t_{12}$ | `[-0.1059, 0.0486] m` | Finite | **VERIFIED ✅** |
| Matrix Orthonormality | $\|R^T R - I\|_\infty < 10^{-6}$ | $< 10^{-5}$ | **VERIFIED ✅** |
| Matrix Determinant | `+1.00000` | $+1.0 \pm 10^{-5}$ | **VERIFIED ✅** |
| Anchor Inlier Count (RANSAC) | `7 / 10` | $\ge 4$ | **VERIFIED ✅** |
| Strict Post-Refit Inliers ($\le 0.20\text{ m}$) | `6 / 10` | $\ge 4$ | **VERIFIED ✅** |
| Consensus RMS Residual | `0.0478 m` (4.78 cm) | $< 0.10\text{ m}$ | **VERIFIED ✅** |
| All-Anchors RMS Residual | `0.1198 m` (11.98 cm) | $< 0.20\text{ m}$ | **VERIFIED ✅** |

---

### 1.2 Entity Cluster Extraction & Intra-Session Consolidation (`vlm/src/cross_session_association.py`)

#### Cluster Extraction Invariants
- **Exact Session Matching:** Replaced loose substring checks (`s1_session in obs_sess`) with exact normalized string equality (`obs_sess == s1_session`), preventing accidental multi-session cross-contamination.
- **Majority Semantic Consensus:** Cluster semantic class is assigned via frequency count across all observations, flagging `semantic_conflict=True` if entropy or minority classes are present.
- **Majority Depth Validity:** Computes `depth_valid_ratio = valid_depth_count / total_observations` and sets `depth_valid = (depth_valid_ratio >= 0.50)`.
- **Optional Spatial Extents:** Avoids synthetic `0.50 m` extents when bounding boxes or MapPoints are absent; tracks `spatial_extent_m: Optional[float] = None` without raising `TypeError`.

#### Complete-Link Intra-Session Consolidation
To prevent transitive $A \leftrightarrow B$ and $B \leftrightarrow C$ chaining:
1. Clusters are merged only if **all pairwise distances between member centroids** satisfy:
   $$\max_{i \in C_a, j \in C_b} \|c_i - c_j\|_2 \le D_{\text{max}} = 0.25\text{ m}$$
2. If adding an adjacent cluster causes the maximum cluster diameter to exceed $0.25\text{ m}$, the merge is rejected.
3. **Mathematically Sound Pooled Uncertainty:**
   Instead of ad-hoc averaging, the consolidated cluster variance $s^2_{\text{pooled}}$ combines within-cluster sample variance and between-cluster centroid dispersion:
   $$s^2_{\text{pooled}} = \frac{1}{K}\sum_{k=1}^K s_k^2 + \frac{1}{K}\sum_{k=1}^K \|\mu_k - \bar{\mu}\|^2$$
   $$\text{SEM}_{\text{pooled}} = \sqrt{\frac{s^2_{\text{pooled}}}{\sum_{k=1}^K N_k}}$$

---

### 1.3 Association Matcher Standardization & Origin-Heuristic Removal

All 5 association matchers adhere to identical input/output contracts without coordinate-origin heuristics:
- **Removed Heuristics:** Eliminated `is_peripheral = (p1[1] < -5.5 or p1[0] < -1.0)`.
- **Principled Anchor Dispersion Gating:** Distance-to-anchor dispersion is now computed relative to the empirical anchor centroid $\bar{c}_{\text{anchor}}$, ensuring rotational and translational equivariance under arbitrary coordinate frame shifts.
- **Graceful Optional Spatial Extent:** Distance computations and bounding box checks test `if c1.spatial_extent_m is not None and c2.spatial_extent_m is not None` before arithmetic.

| Matcher Name | Algorithmic Formulation | Precision | Static Recall | Reloc Recall | Instance F1 |
|---|---|:---:|:---:|:---:|:---:|
| **Baseline 1: Hungarian Bipartite** | Global Munkres assignment on Euclidean centroid distances | 42.11% | 15.38% | 12.50% | 18.75% |
| **Baseline 2: Greedy Nearest Neighbor** | Greedy local minimum Euclidean distance matching | 40.00% | 15.38% | 12.50% | 18.75% |
| **Baseline 3: Spatial Proximity** | Fixed radius heuristic ($R = 0.50\text{ m}$) | 35.71% | 19.23% | 0.00% | 20.83% |
| **Baseline 4: Naive Overlap** | Extent-based bounding box intersection without uncertainty | 28.57% | 15.38% | 0.00% | 16.67% |
| **Proposed: Uncertainty-Adaptive** | Complete-link clusters + 3-sigma gate + Mahalanobis gating | **53.85%** | **23.08%** | **25.00%** | **29.17%** |

---

### 1.4 Decoupling Predictions from Ground Truth & Metric Disambiguation

#### Decoupled Identity Formulation
In `vlm/scripts/run_full_experiment.py`, persistent entity assignment in prediction mode now uses system-generated identifiers:
- Predicted persistent IDs: `matched_m.persistent_entity_id` (e.g. `cluster_001_matched`).
- Ground-truth target annotations are retained exclusively for offline evaluation scoring and never leak into the scene graph node creation or state estimation logic.

#### Disambiguation of False Split and Miss Rates
In `evaluate_matcher_against_ground_truth`:
- **`instance_miss_rate_pct` (formerly misnamed `false_split_rate`):**
  $$\text{Miss Rate} = \frac{N_{\text{gt\_instances}} - N_{\text{recovered}}}{N_{\text{gt\_instances}}} \times 100\% = 70.83\%$$
- **`true_false_split_rate_pct`:**
  Percentage of ground-truth physical instances that are split across $\ge 2$ distinct predicted persistent entity IDs:
  $$\text{True False-Split Rate} = \frac{|\{inst : |\{pred\_id(trk) : trk \in inst\}| \ge 2\}|}{N_{\text{gt\_evaluable}}} \times 100\% = 0.0\%$$
- **`true_false_merge_rate_pct`:**
  Percentage of predicted persistent entities that encompass $\ge 2$ distinct ground-truth physical instances:
  $$\text{True False-Merge Rate} = \frac{|\{pred : |\{gt(trk) : trk \in pred\}| \ge 2\}|}{N_{\text{pred\_evaluable}}} \times 100\% = 0.0\%$$

---

## Part 2: Exhaustive Audit of All 100%, 0%, and Near-100% Metrics

### 2.1 Depth Validity Rate (Reported: 100.0%)

- **Measurement Type:** Sensor Depth Presence.
- **Evaluated Scope:** 7,956 / 7,956 evidence records.
- **Denominator:** 7,956 total observations.
- **Integrity Finding:**
  The `100.0%` figure reflects **measurement presence**: every observation accepted into the pipeline has a valid, non-null, finite positive sensor depth reading within the sensor operational range $[0.4\text{ m}, 6.0\text{ m}]$.
- **Depth Quality Audit:**
  Depth quality is not binary; it is characterized by empirical depth uncertainty and point-to-bounding-box geometric consistency:
  - Mean 1-sigma depth uncertainty $\bar{\sigma}_z = 0.0242\text{ m}$ (2.42 cm).
  - Mean point-to-bbox inlier ratio $= 94.62\%$.
  - Median point-to-bbox distance outside box $= 0.0\text{ px}$.
- **Resolution:** Retained `depth_validity_percentage = 100.0%` for backward compatibility, added explicit `depth_presence_rate_pct = 100.0%`, and clarified in all documentation that this represents sensor measurement presence rather than sub-millimeter depth accuracy.

---

### 2.2 False Motion Rate (0.0%) & Static Stability Pass Rate (100.0%)

- **Measurement Type:** Static Structure Specificity.
- **Evaluated Scope:** 7 verified physical static fixtures:
  `["desk_001", "desk_018", "desk_025", "desk_031", "door_001", "ceiling_fan_001", "wall_boundary_north"]`
- **Denominator:** 7 static entities ($N=7$).
- **Integrity Finding:**
  Across all multi-view observations of these 7 structural objects, persistent displacement between Session 1 and Session 2 remained strictly within the 3-sigma gate ($0.12\text{ m}$). Zero of the 7 structures were classified as moving.
- **Resolution:**
  - Added explicit fields: `static_structures_evaluated_count = 7`, `static_structures_falsely_moving_count = 0`.
  - Renamed the metric to `static_stability_pass_rate_pct = 100.0%`, retaining `static_stability_accuracy_pct = 100.0%` as an alias.
  - Documented that this measures static fixture specificity on the 7 architectural fixtures, not whole-dataset trajectory accuracy.

---

### 2.3 Occlusion Recovery Rate (Formerly Reported: 100.0% $\rightarrow$ Corrected: `NOT_EVALUABLE`)

- **Prior Reported Value:** `100.0%`
- **Prior Code:**
  ```python
  "occlusion_recovery_rate_pct": 100.0 if occlusion_events > 0 else 0.0
  ```
- **Integrity Finding:**
  This metric was an unprincipled placeholder that returned `100.0%` whenever `occlusion_events > 0`. There was no algorithm verifying whether the occluded track was re-identified after the gap versus assigned a new ID.
- **Resolution:**
  - Removed the fabricated formula.
  - Set `"occlusion_recovery_rate_pct": null`.
  - Added explicit diagnostic status: `"occlusion_recovery_status": "NOT_EVALUABLE_NO_GT_REID"`.
  - Documented that authentic cross-occlusion re-identification requires ground-truth continuous camera poses and re-ID annotations, which are not captured in the upstream bounding-box stream.

---

### 2.4 Semantic Macro-F1 (Reported: 100.0% on Matched Support)

- **Measurement Type:** Conditional Taxonomy Prior Baseline.
- **Evaluated Scope:** 3,710 / 7,956 observations (46.63% evaluation coverage).
- **Excluded Scope:** 4,246 / 7,956 observations (53.37% out-of-vocabulary or unmapped detector labels).
- **Evaluated Classes:** `IMMOVABLE` (support 1,842), `MOVABLE` (support 1,858), `MOBILE` (support 10).
- **Integrity Finding:**
  The `100.0%` Macro-F1 score is conditioned strictly on observations whose raw detector labels map directly into the 3 evaluated taxonomy classes. It evaluates the deterministic consistency of the fixed class-mobility prior lookup table (e.g., `desk -> IMMOVABLE`, `chair -> MOVABLE`, `robot -> MOBILE`).
- **Resolution:**
  - Added explicit metadata: `"evidence_type": "fixed_semantic_class_mobility_prior_baseline"`.
  - Added `"evaluable_coverage_ratio": "3710/7956"`.
  - Added `"evaluation_scope": "Matched ground-truth taxonomy support only"`.
  - Added prominent warnings in the Markdown evidence report prohibiting citing this score as whole-pipeline classification accuracy.

---

### 2.5 Track Fragmentation Rate (Reported: 7.14% vs 82.44%)

- **Measurement Type:** Physical Instance Fragmentation vs Raw Upstream Detector Transience.
- **Dual Reporting:**
  1. **Ground-Truth Instance Fragmentation (`7.14%`):**
     Evaluates whether ground-truth physical objects are split into multiple tracks within the same session. Of 28 evaluable physical objects, 2 exhibited multiple tracks within a single session ($2 / 28 = 7.14\%$).
  2. **Raw Detector Transient Track Rate (`82.44%`):**
     Evaluates upstream detector stability. Out of 1,970 raw detector tracks, 1,624 persisted for $\le 4$ frames ($1,624 / 1,970 = 82.44\%$).
- **Integrity Finding:**
  Both numbers are authentic, but they measure distinct layers of the perception hierarchy: the upstream 2D bounding-box detector suffers from high transient fragmentation ($82.44\%$), which the downstream spatial-temporal clustering successfully consolidates into low physical-instance fragmentation ($7.14\%$).
- **Resolution:** Both perspectives are explicitly exposed with unambiguous labels in all output summaries and reports.

---

## Part 3: Verification & Test Suite Results

The entire perception codebase was verified using the project Python environment:

```powershell
& "C:\Users\Yash Bhardwaj\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m pytest vlm/tests/ --ignore=vlm/tests/_test_run_tmp --ignore=vlm/tests/basetemp_accessible --ignore=vlm/tests/basetemp_ed318c66 --ignore=vlm/tests/pytest_basetemp --ignore=vlm/tests/test_tmp_hk162rjr -p no:cacheprovider
```

### Test Suite Summary
- **Total Tests Collected:** 154
- **Passed:** 154
- **Failed:** 0
- **Pass Rate:** **100.0%**
- **Execution Time:** 6.46 seconds

### Pipeline Verification Scripts
1. `vlm/scripts/evaluate_stage5_evidence.py`: Passed (7,956 records processed, acceptance gates passed).
2. `vlm/scripts/audit_semantic_associations.py`: Passed (keyed joins, canonical paths, summary generated).
3. `vlm/scripts/evaluate_tracklet_consolidation.py`: Passed (1,970 raw tracks audited, consolidation complete).
4. `vlm/scripts/run_full_experiment.py`: Passed (reproducible end-to-end execution in 7.14s, scene graph synthesized).
5. `vlm/scripts/validate_research_artifacts.py`: Passed (100% artifact and metric integrity verified).
6. `vlm/scripts/generate_research_tables.py`: Passed (all tables regenerated across CSV, Markdown, LaTeX).

---

## Part 4: Deliverables & Audit Manifest

| Artifact File | Description | Verification Status |
|---|---|:---:|
| [stage5_adapter.py](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/src/stage5_adapter.py) | Strict registration validation, zero fallback constants, SVD rank check | **VERIFIED ✅** |
| [cross_session_association.py](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/src/cross_session_association.py) | Complete-link consolidation, pooled SEM, metric decoupling, 5 standardized matchers | **VERIFIED ✅** |
| [run_full_experiment.py](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/scripts/run_full_experiment.py) | Decoupled persistent entity IDs, isolated evidence directory, verified registration | **VERIFIED ✅** |
| [evaluate_stage5_evidence.py](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/scripts/evaluate_stage5_evidence.py) | Elimination of 100% occlusion recovery, presence vs quality, static specificity | **VERIFIED ✅** |
| [audit_semantic_associations.py](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/scripts/audit_semantic_associations.py) | Canonical observation resolution, keyed joins on `observation_id` | **VERIFIED ✅** |
| [metric_integrity_audit.json](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/outputs/metric_integrity_audit.json) | Structured machine-readable audit report of all invariants, matchers, and metrics | **VERIFIED ✅** |
| [metric_integrity_audit.md](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/outputs/metric_integrity_audit.md) | Exhaustive human-readable audit and integrity documentation | **VERIFIED ✅** |
