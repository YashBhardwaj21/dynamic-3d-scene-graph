# Data and Evaluation Inventory: Populations, Benchmarks, and Caches

**Document Version:** `1.0.0`  
**Audit Date:** `2026-10-16`  
**Repository:** `YashBhardwaj21/dynamic-3d-scene-graph`

This document provides an exhaustive, authoritative inventory of all data populations, benchmark manifests, physical-instance splits, multi-view image assets, evaluation caches, and ground-truth artifacts within the repository.

---

## 1. Audit of the 10 Headline Quantities

Each reported quantity from the repository audit has been independently re-evaluated and verified against current checkout files:

| Quantity | Historical Claim | Recomputed Value | Exact Source File & Lineage | Verification Status | Scientific Integrity / Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Raw Observations** | 7,956 records | **7,956 records** | `vlm/outputs/object_observations.jsonl` | `VERIFIED` | 100% valid JSON records across 2 SLAM sessions (`recording_01`, `recording_02`). |
| **2. Unique Tracker Tracks** | 1,385 tracks | **1,385 tracks** | `vlm/outputs/object_observations.jsonl` (`local_track_id`) | `VERIFIED` | Produced by Kalman-filter tracker; high count reflects tracking loss. |
| **3. Track Fragmentation Rate** | 72.78% ($\le 4$ frames) | **72.78% (1,008 tracks)** | `vlm/outputs/stage5_evidence_metrics.json` | `VERIFIED` | Major bottleneck: 1,008 tracks survive $\le 4$ frames, degrading centroid estimates. |
| **4. Physical Ground-Truth Instances** | 52 instances | **52 instances** | `vlm/benchmarks/movability_ground_truth.json` | `VERIFIED` | Inter-annotator agreement: 96.15% (Cohen's $\kappa = 0.8587$). Severe class imbalance (45 MOVABLE, 4 IMMOVABLE, 3 MOBILE). |
| **5. Multi-View Crop Instances** | 39 instances | **39 instances (273 crops)** | `vlm/outputs/multiview_samples/` | `VERIFIED` | 39 instance directories containing 1 to 3 camera angles (front, side, detail) + scene context. |
| **6. Stage 6 Router Manifest** | 15 instances | **15 instances** | `vlm/benchmarks/router_evaluation_manifest.json` | `VERIFIED` | Balanced development split: 5 IMMOVABLE, 5 MOVABLE, 5 MOBILE. |
| **7. Held-Out Affordance Set** | 28 instances | **28 instances** | `vlm/outputs/multiview_affordance_results.json` | `VERIFIED` | Single-class furniture subset: **28 MOVABLE, 0 IMMOVABLE, 0 MOBILE**. 100% score is a single-class artifact! |
| **8. Semantic Evaluation Exclusion** | 44.43% unmapped | **44.43% (3,535 records)** | `vlm/outputs/stage5_evidence_metrics.json` | `VERIFIED` | Out of 7,956 records, only 4,421 match the 3 taxonomy classes. 99.76% Macro-F1 only covers 55.57% of data. |
| **9. Association Physical Recall** | 23.33% recall | **23.33% (7 / 30)** | `vlm/outputs/full_experiment_run/association_baselines_summary.json` | `VERIFIED` | Pinned across all 4 baselines due to 17 gate rejections from noisy track centroids. |
| **10. Controlled Relocation Error**| 30.42% relative error | **30.42% (0.1247m mean)** | `vlm/outputs/stage5_evidence_metrics.json` | `VERIFIED` | 3 tape-measured cases: Chair 002 (9.52%), Chair 003 (32.34%), Box 007 (49.42%). |

---

## 2. Comprehensive Population Taxonomy & Split Overlap Analysis

```
[Total Raw Observations: 7,956 records (1,385 unique tracks)]
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
[In-Vocabulary Subset: 4,421 records]  [Unmapped Out-of-Vocabulary: 3,535 records]
(Macro-F1 = 99.76% on this subset)      (Excluded from historical reports)
```

```
[Physical Ground Truth Universe: 52 verified instances]
   ├── Category: IMMOVABLE (4), MOVABLE (45), MOBILE (3)
   ├── Cross-Session Multi-Session Instances: 30 instances (26 static, 4 relocated)
   │     └── Cross-Session Association Benchmark evaluated on these 30 instances
   │
   └── Multi-View Image Harvested Universe: 39 instances (in multiview_samples/)
         ├── Stage 6 Router Manifest: 15 instances (5 IMMOVABLE, 5 MOVABLE, 5 MOBILE)
         │     └── 11 instances overlap with the 39 harvested instances
         │
         └── Stage 7 "Held-Out" Set: 28 instances (39 minus 11)
               └── 100% MOVABLE furniture (15 chairs, 10 desks, 2 boxes, 1 backpack)
```

### Split Overlap & Leakage Evaluation
- **No Test-to-Train Data Leakage:** The router evaluation manifest (15 instances) and held-out affordance set (28 instances) are strictly disjoint at the physical instance level.
- **Physical Instance Discrepancy:** The full physical universe contains 52 instances, but only 39 have harvested multi-view image crops. The remaining 13 instances lack RGB crops because they were observed exclusively in peripheral frames where bounding box quality was degraded.

---

## 3. Evaluation Caches & Historical Artifacts

### 3.1 VLM Response Cache (`vlm/outputs/stage6/router_vlm_cache.json`)
- **Entries:** 15 records corresponding to the Stage 6 evaluation instances.
- **Response Validity:** 100.0% valid JSON schemas.
- **Response Distribution:**
  - `IMMOVABLE`: 3
  - `MOVABLE`: 9
  - `MOBILE`: 3
- **Latency Accounting:** Historical records recorded cached replay timings ($\approx 0.005$ ms). Live inference timings require live execution.

### 3.2 Registration Audit Cache (`vlm/outputs/stage5_registration_audit.json`)
- **Total Anchors:** 10 static architectural correspondences.
- **Robust Inliers:** 7 anchors with inlier RMS residual = $0.1198$ m.
- **Boundary Outliers:** 3 anchors with residuals up to $0.6180$ m.
- **Cross-Validation:** 5-fold cross-validation yields mean fold RMS = $0.3021$ m and pooled RMSE = $0.3538$ m.

### 3.3 Dynamic Scene Graph Output (`vlm/outputs/full_experiment_run/scene_graph.json`)
- **Total Nodes:** 195 fused 3D entities.
- **Stationary Fused Entities:** 29 nodes.
- **Relocated Fused Entities:** 12 nodes (preserving dual pose history).
- **Invalidated SLAM Landmarks:** 12 feature points explicitly flagged for landmark de-weighting.
