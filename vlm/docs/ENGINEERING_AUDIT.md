# Engineering Audit: Dynamic 3D Scene Graph VLM Subsystem

**Document Version:** `1.0.0`  
**Audit Date:** `2026-10-16`  
**Repository:** `YashBhardwaj21/dynamic-3d-scene-graph`  
**Git Commit SHA:** `e9c336423c726d400e9603f9f3152dc27498e72c`  
**Branch:** `main` (Working Tree Clean)  
**Governing Research Question:**  
> *Does estimator-aware selective VLM reasoning improve the reliability of SLAM observations in changing indoor environments compared with strong geometry, temporal, semantic, and non-estimator-aware routing baselines, while reducing unnecessary VLM inference?*

---

## 1. Initial State Assessment & Repository Reconciliation

### 1.1 Working Tree & Git State
- **Branch:** `main`
- **Head Commit:** `e9c3364` (`refactor: organized vlm repo`)
- **Working Tree:** 100% clean, zero modified files, zero unstaged changes.
- **Assignment Bundle:** The self-contained non-destructive assignment directory `vlm/assignment/` was committed in `e9c3364`, containing 346 cryptographically verified files, research-quality figures (SVG/PNG), documentation, and an AI review prompt.
- **Active Tests:** The authoritative test suite at `vlm/tests/` passes 99/99 pytest tests in ~6.8 seconds (`pytest vlm/tests -q`).

### 1.2 Upstream & Downstream Context
The repository consists of:
1. `src/scene_graph/`: Upstream ROS 2 nodes, spatial tracking, geometry, navigation, and temporal clustering infrastructure.
2. `ros2_ws/`: ROS 2 packages (`d455_bridge`, `scene_graph_ros`) interfacing Intel RealSense D455 depth camera and RTAB-Map SLAM.
3. `vlm/`: The core research workspace containing evidence extraction (`vlm/src/`), evaluation and ablation runners (`vlm/scripts/`), benchmark annotations (`vlm/benchmarks/`), experimental metrics (`vlm/outputs/`), and test suites (`vlm/tests/`).

---

## 2. End-to-End Architectural Pipeline Trace

The VLM subsystem operates across six sequential stages:

```
[Raw Observations: object_observations.jsonl (7,956 records)]
                           │
                           ▼
      [Stage 5 Adapter: vlm/src/stage5_adapter.py]
                           │
                           ▼
      [Stage 5 Pipeline: vlm/src/stage5_pipeline.py]
         ├── Geometric Evidence: geometric_evidence.py
         ├── Temporal Evidence: temporal_evidence.py
         └── Semantic Evidence: semantic_evidence.py
                           │
                           ▼
      [Evidence Fusion: vlm/src/evidence_fusion.py]
                           │
                           ▼
  [Inter-Session Registration: cross_session_association.py]
      (Robust SE(2) SVD on 10 Stationary Desks & Boxes)
                           │
                           ▼
  [Cross-Session Data Association: cross_session_association.py]
      (Centroid, Hungarian, Uncertainty-Gated, Relocation-Aware)
                           │
                           ▼
    [Selective VLM Router: selective_router.py & router_policy.py]
      (Fast Rule Filter vs Offline-Cached Multiview VLM Queries)
                           │
                           ▼
  [Contextual Scene Graph Fusion: contextual_scene_graph_fusion.py]
      (195 Nodes, 29 Stationary Merged, 12 Relocated, 12 Invalidated Landmarks)
```

### 2.1 Field-by-Field Evidence Provenance Matrix

| Field | Source Module & Function | Computation Method | Provenance Classification | Failure Mode / Caveat |
| :--- | :--- | :--- | :--- | :--- |
| `centroid_floor_m` | `stage5_adapter.py`: `_extract_centroid_floor()` | 2D projection $[x, y]$ of 3D bounding box center transformed by RTAB-Map camera pose $T_{world \leftarrow cam}$. | `Derived from Measured` | Centroid jitters by up to 0.196m due to 2D detector bounding box edge instability. |
| `extent_m` | `stage5_adapter.py`: `_extract_extent()` | Diagonal extent $\|p_{max} - p_{min}\|$. | `Derived from Measured` | Subject to bounding box noise and partial occlusions. |
| `depth_validity` | `geometric_evidence.py`: `compute_geometric_evidence()` | Checks if sensor depth values exist inside projected 3D box. | `Measured` | Reported as 100.0% valid across 7,956 records. |
| `point_to_bbox_px` | `geometric_evidence.py` | 2D reprojection distance between projected 3D vertices and 2D detection box. | `Derived from Measured` | Mean distance 5.43 px, p95 14.88 px. |
| `lifespan_frames` | `temporal_evidence.py`: `compute_temporal_evidence()` | Total consecutive frames observation track survived. | `Measured` | **72.78% of tracks span $\le 4$ frames** (1,008 / 1,385), causing extreme fragmentation. |
| `occlusion_events` | `temporal_evidence.py` | Tracks re-acquired after visual interruption. | `Measured` | 355 events recorded; 100.0% recovery reported. |
| `semantic_class` | Detector / `semantic_evidence.py` | Zero-shot CLIP and YOLO detector output mapped to 3 classes (`IMMOVABLE`, `MOVABLE`, `MOBILE`). | `Inferred` | **44.43% of records (3,535 / 7,956) have unmapped detector labels.** |
| `semantic_fused_f1`| `semantic_evidence.py` | Macro-F1 across matched records. | `Derived from Inferred` | Reported as 99.76%, but only covers 55.57% of total records. |
| `se2_transform` | `cross_session_association.py`: `compute_robust_se2()` | RANSAC SVD alignment on 10 stationary anchors. | `Derived from Measured` | Inlier RMS = 0.1198m, but boundary anchors suffer 0.618m residual drift. |
| `cross_session_pairs` | `cross_session_association.py`: `associate_cross_session_candidates()` | Gated bipartite matching across recording sessions. | `Inferred` | **Recall pinned at 23.33%** (7 / 30) due to fragmented track centroids exceeding 0.45m gate. |
| `vlm_decision` | `selective_router.py`: `route_instance()` | Cost-benefit policy selecting cheap resolution or VLM escalation. | `Inferred` | Uses rule-based heuristics and offline VLM cache; M3 all-VLM baseline achieves only 33.33% balanced accuracy. |
| `slam_invalidated` | `contextual_scene_graph_fusion.py`: `fuse_scene_graph()` | Flags relocated object landmarks ($\ge 0.30$m shift) as invalid for SLAM. | `Derived from Inferred` | 12 visual landmarks invalidated in full experiment run. |

---

## 3. Ground-Truth Data & Benchmark Lineage

The repository contains five distinct evaluation populations that must never be conflated:

1. **Total Sensory Records (N = 7,956 observations):** Raw bounding boxes and RTAB-Map poses from `semantic_mapping_my_desk_recording_01` and `recording_02`.
2. **Front-End Tracker Population (N = 1,385 unique tracks):** Causal Kalman-filtered tracks produced by the detector front-end.
3. **Consolidated Physical Ground Truth (N = 52 physical instances):** Adjudicated multi-annotator benchmark in `movability_ground_truth.json` (4 IMMOVABLE, 45 MOVABLE, 3 MOBILE).
4. **Multi-View Crop Collection (N = 39 physical instances):** 273 harvested multi-view RGB image crops in `vlm/outputs/multiview_samples/`.
5. **Stage 6 Router Benchmark Manifest (N = 15 physical instances):** Balanced development manifest in `router_evaluation_manifest.json` (5 IMMOVABLE, 5 MOVABLE, 5 MOBILE).
6. **Held-Out Affordance Set (N = 28 physical instances):** Strict non-manifest subset of the 39 instances (all 28 are `MOVABLE` indoor furniture).
7. **Cross-Session Association Benchmark (N = 30 multi-session physical instances):** 26 static instances + 4 relocated instances with tracks in both sessions.
8. **Controlled Relocation Ground Truth (N = 3 instances):** Tape-measured physical relocation events (`chair_002`, `chair_003`, `storage box_007`).

---

## 4. Key Architectural Findings & Deficiencies

1. **No End-to-End Estimator-Aware Routing Implementation:**  
   `router_policy.py` implements a heuristic rule-based confidence policy based on semantic prior lookup and static spatial stability. It does not measure the actual *influence* of an observation on SLAM state estimation or trajectory loss.
2. **Visual Input Truncation:**  
   `selective_router.py` strictly caps visual inputs to at most two composite views (`crop_view_0`, `scene_view_0`, `crop_view_1`, `scene_view_1`) and assumes views are consecutive pairs. It lacks deterministic multi-view selection, hashing, and decoding failure rejection.
3. **Severe Upstream Tracker Fragmentation:**  
   72.78% of front-end tracks survive $\le 4$ frames. Spatial gates in `cross_session_association.py` reject 17 of 30 physical instances because fragmented track centroids have not converged, bottlenecking instance recall at 23.33%.
4. **Misleading Evaluation Reporting:**  
   Historical reports tout a 99.76% semantic Macro-F1, failing to report that 44.43% of the dataset was unmapped and excluded from the evaluation.
5. **Single-Class Held-Out Set:**  
   The 28-instance held-out affordance test split consists entirely of `MOVABLE` furniture, making 100% accuracy on that split a trivial single-class artifact rather than proof of generalized multi-class reasoning.
