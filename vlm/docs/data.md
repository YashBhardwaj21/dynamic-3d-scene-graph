# Section 3 Raw-Sequence Dataset Audit & Problem Formulation Report

**Document File:** `vlm/docs/data.md`  
**Audit Date:** `2026-10-18`  
**Governing Workspace:** `c:\Users\Yash Bhardwaj\Desktop\perception`  
**Target Paper Section:** Section 3 — *Problem Formulation: Estimator-Aware Dynamic 3D Scene Graphs*  
**Operational Mode:** Read-Only Audit & Formal State Contract Reconciliation  

---

## 1. Executive Findings (Ranked by Severity)

1. **Critical Prediction/Evaluation Contamination in Experiment Orchestration**  
   *Location:* [`vlm/scripts/run_full_experiment.py:223-230`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/scripts/run_full_experiment.py#L223-L230), [`L255`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/scripts/run_full_experiment.py#L255), [`L275-281`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/scripts/run_full_experiment.py#L275-L281), [`L338`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/scripts/run_full_experiment.py#L338), [`L384-425`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/scripts/run_full_experiment.py#L384-L425)  
   The experiment orchestrator directly loads ground-truth evaluation artifacts ([`controlled_scene_events.json`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/benchmarks/controlled_scene_events.json) and [`cross_session_identity_annotations.json`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/benchmarks/cross_session_identity_annotations.json)) during inference scene graph creation. For policies R3 and R4, the script explicitly overrides `matched_node.slam_invalidated = True`, forces `matched_node.observed_motion = ObservedMotionState.DISPLACED`, and overwrites `matched_node.centroid_world` with `ev.get("final_position_floor_m")`. The resulting scene graph is not a pure prediction from sensory evidence.

2. **Downstream Results Inherit Contaminated Orchestrator Artifacts**  
   *Location:* [`vlm/outputs/downstream_slam_effects.json:5-6`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/outputs/downstream_slam_effects.json#L5-L6), [`L23-33`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/outputs/downstream_slam_effects.json#L23-L33)  
   The downstream validation report evaluates `vlm/outputs/full_experiment_run/scene_graph.json`, which was produced by the contaminated orchestrator. The reported 25.0% relocation invalidation rate (1/4 correct: detecting `inst_human_operator_001` while missing `chair_002`, `chair_003`, and `storage_box_007`) is an artifact of the contaminated pipeline and association gate rejections, not a clean sensor-driven baseline.

3. **Multi-Run Manifest Inconsistency (180 Nodes vs. 195 Nodes)**  
   *Location:* [`vlm/outputs/full_experiment_run/run_manifest.json:5`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/outputs/full_experiment_run/run_manifest.json#L5), [`L41`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/outputs/full_experiment_run/run_manifest.json#L41), [`vlm/outputs/downstream_slam_effects.json:6`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/outputs/downstream_slam_effects.json#L6)  
   The committed artifacts in `vlm/outputs/` stem from disparate execution revisions:
   - `run_manifest.json` was generated under commit `e625ea2` and logs **180 fused scene graph nodes** (30 stationary, 11 relocated, 38 invalidated).
   - `downstream_slam_effects.json` evaluates a **195-node scene graph** (45 stationary, 4 relocated, 12 invalidated) from a different run.
   These artifacts cannot be presented as a single coherent experimental run.

4. **Broken Observation Lookup Index Producing Universal Timestamp Fallbacks**  
   *Location:* [`vlm/scripts/run_full_experiment.py:218`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/scripts/run_full_experiment.py#L218)  
   The orchestrator constructs its observation lookup index using `obs_by_id = {o.get("observation_id"): o for o in observations if o.get("observation_id")}`. However, `vlm/data/observations/object_observations.jsonl` strictly uses the key `"obs_id"`. Because `"observation_id"` is never present, `obs_by_id` is perpetually empty (`{}`). Every entity node in the fused scene graph silently falls back to synthetic hardcoded timestamps (`100.0` for Session 1, `200.0` for Session 2), corrupting temporal velocity and lifespan tracking.

5. **Absence of Independent 6-DoF Camera Pose Ground Truth**  
   *Location:* [`vlm/data/raw/my_desk_recording_01/groundtruth.txt:1-3`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/data/raw/my_desk_recording_01/groundtruth.txt#L1-L3), [`vlm/outputs/downstream_slam_effects.json:173-187`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/outputs/downstream_slam_effects.json#L173-L187)  
   `groundtruth.txt` in both sessions contains identity rotations (`0 0 0 1`) and zero translations (`0 0 0`), explicitly annotated as `# Camera trajectory (SYNTHETIC IDENTITY PLACEHOLDER - NO GROUND TRUTH AVAILABLE)`. No optical motion capture (mocap) system was recorded. Consequently, `CameraTrajectory.txt` is an estimated SLAM output (ORB-SLAM3/RTAB-Map), not an independent ground truth. Metric claims regarding Absolute Trajectory Error (ATE) or Relative Pose Error (RPE) cannot be substantiated without external ground-truth trajectories.

6. **Active Registration Parameter Discrepancy**  
   *Location:* [`vlm/outputs/stage5_registration_audit.json:4-8`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/outputs/stage5_registration_audit.json#L4-L8), [`vlm/data/semantic_mapping_fused_desk_run/session_registration.json:4-8`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/data/semantic_mapping_fused_desk_run/session_registration.json#L4-L8)  
   Two conflicting registration files exist:
   - **Authoritative Orchestrator Input:** [`vlm/outputs/stage5_registration_audit.json`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/outputs/stage5_registration_audit.json) (SHA-256: `d6ce09f4...`), with $\theta = -15.558^\circ$, $\mathbf{t} = [-0.1059, 0.0486]^\top\text{ m}$, inlier RMS = $0.1198\text{ m}$, anchor RMS = $0.3087\text{ m}$.
   - **Legacy Mapping Output:** [`vlm/data/semantic_mapping_fused_desk_run/session_registration.json`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/data/semantic_mapping_fused_desk_run/session_registration.json), with $\theta = -16.13^\circ$, $\mathbf{t} = [-0.0815, 0.0930]^\top\text{ m}$, anchor RMS = $0.3054\text{ m}$.
   All analytical displacement calculations must strictly cite the active audit file (`-15.558°`, `[-0.1059, 0.0486] m`).

7. **Ontology Citation and Label Space Disconnect**  
   *Location:* [`vlm/benchmarks/movability_ground_truth.json:8-16`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/benchmarks/movability_ground_truth.json#L8-L16), [`L38-42`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/benchmarks/movability_ground_truth.json#L38-L42)  
   The historical reference associating the movability ontology with "arXiv:2404.09904" is erroneous (resolving to an unrelated condensed-matter physics paper). Specific external attribution is removed; the taxonomy is formulated as this project's operational ontology. Furthermore, benchmark ground-truth labels record only a 3-state motion space (`STATIC`, `MOVING`, `UNCERTAIN`). The proposed 4-state space (`STATIC`, `DISPLACED`, `DYNAMIC`, `UNCERTAIN`) is a problem formulation target, not a fully annotated benchmark label space.

8. **Mischaracterization of `object_tracks.jsonl`**  
   *Location:* [`vlm/data/observations/object_tracks.jsonl`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/data/observations/object_tracks.jsonl)  
   `object_tracks.jsonl` contains 52 rows representing physical instance summaries with ground-truth fields (`object_instance_id`, `movability_gt`, `current_motion_gt`, `associated_track_ids`). It is **not** a raw Kalman-filter tracker stream. The 1,385 count represents unique `local_track_id` values found in `object_observations.jsonl`, not rows in `object_tracks.jsonl`.

9. **Inappropriate Default Observed Motion in Persistent Nodes**  
   *Location:* [`vlm/src/contextual_scene_graph_fusion.py:62-67`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/src/contextual_scene_graph_fusion.py#L62-L67)  
   `ContextualSceneGraphNode` initializes `observed_motion = ObservedMotionState.STATIC` by default even when `geometry_valid = False`. Incomplete, unobserved, or unverified entities automatically default to being confident static landmarks, violating fail-closed epistemic safety.

10. **Replay Simulation Confused with Live VLM Inference Reduction**  
    *Location:* [`vlm/outputs/downstream_slam_effects.json:188-196`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/outputs/downstream_slam_effects.json#L188-L196)  
    The reported reduction (3 queries in R3 vs. 15 in R4) and 3.67x latency speedup reflect router decision rules evaluated against an offline response cache (`router_vlm_cache.json`) on the 15-instance development manifest (`router_evaluation_manifest.json`), with zero live API dispatches. It does not establish live end-to-end model execution speedup.

---

## 2. Repository and Raw-Data Inventory

### 2.1 File Inventory and Provenance Matrix

| Directory / File Path | Status | Count / Size | Provenance / Generator | Role in Architecture & Section 3 Limitations |
| :--- | :--- | :--- | :--- | :--- |
| `data/raw/my_desk_recording_01/rgb/` | Local on disk | 5,207 PNGs (2.05 GB) | RealSense D455 capture | Raw 8-bit RGB frames ($640 \times 480$). **Not tracked in GitHub tree.** |
| `data/raw/my_desk_recording_01/depth/` | Local on disk | 5,271 PNGs (1.28 GB) | RealSense D455 IR stereo | Raw 16-bit depth PNGs (scale = 0.001 m/unit). **Not tracked in GitHub tree.** |
| `data/raw/my_desk_recording_02/rgb/` | Local on disk | 5,287 PNGs (2.09 GB) | RealSense D455 capture | Raw 8-bit RGB frames ($640 \times 480$). **Not tracked in GitHub tree.** |
| `data/raw/my_desk_recording_02/depth/` | Local on disk | 5,288 PNGs (1.29 GB) | RealSense D455 IR stereo | Raw 16-bit depth PNGs (scale = 0.001 m/unit). **Not tracked in GitHub tree.** |
| `vlm/data/raw/my_desk_recording_01/associations.txt` | Tracked | 5,108 pairs (464.8 KB) | `associate.py` (nominal $\Delta t \le 0.02\text{ s}$; 91.9% $\le 20\text{ ms}$, max 46.254 ms) | RGB-to-depth timestamp pairing for Session 1. |
| `vlm/data/raw/my_desk_recording_02/associations.txt` | Tracked | 5,283 pairs (475.5 KB) | `associate.py` (nominal $\Delta t \le 0.02\text{ s}$; 97.2% $\le 20\text{ ms}$, max 22.140 ms) | RGB-to-depth timestamp pairing for Session 2. |
| `vlm/data/raw/my_desk_recording_01/camera_intrinsics.json` | Tracked | 449 B | RealSense factory EEPROM | Pin-hole model with Inverse Brown-Conrady coefficients ($f_x=383.92, f_y=383.48$). |
| `vlm/data/raw/my_desk_recording_01/groundtruth.txt` | Tracked | 5,210 lines (354.2 KB) | **Synthetic Placeholder** | **Identity poses only (`0 0 0 0 0 0 1`). Zero valid camera trajectory ground truth.** |
| `vlm/data/semantic_mapping_my_desk_recording_01/trajectory/CameraTrajectory.txt` | Tracked | 5,108 poses (537.2 KB) | ORB-SLAM3 RGB-D mode | Estimated camera poses in `slam_world` frame. **Atlas unverified.** |
| `vlm/data/semantic_mapping_my_desk_recording_02/trajectory/CameraTrajectory.txt` | Tracked | 5,283 poses (556.5 KB) | ORB-SLAM3 RGB-D mode | Estimated camera poses in `slam_world` frame. **Atlas unverified.** |
| `vlm/data/observations/object_observations.jsonl` | Tracked | 7,956 records (6.48 MB) | Front-end detector + SLAM projection | Frame-by-frame 3D bounding boxes (S1: 2,934; S2: 5,022). **Uses `"obs_id"` and `"ESTIMATED_SLAM_TRAJECTORY"`.** |
| `vlm/data/observations/object_tracks.jsonl` | Tracked | 52 records (84.8 KB) | Benchmark Physical Summary | **52 physical instance records with ground truth fields.** Not raw tracking stream. |
| `vlm/data/multiview_samples/` | Tracked | 53 dirs, 273 crops | Harvested view crops | 39 verified instances with multi-view crops; 13 instances have zero views. |
| `vlm/benchmarks/movability_ground_truth.json` | Tracked | 52 instances (39.8 KB) | Dual-annotator adjudication | Adjudicated movability ground truth ($\kappa = 0.8587$). **Uses 3-state motion space.** |
| `vlm/benchmarks/controlled_scene_events.json` | Tracked | 6 events (3.7 KB) | Calibrated steel tape measure | 3 controlled relocations (`chair_002`, `chair_003`, `storage_box_007`), 1 robot run. |

---

## 3. Sequence Validation, Timing, Calibration, and Coordinate Frames

### 3.1 Sequence Metric Reconciliation Table

| Metric | Session 1: Raw Meta | Session 1: Validation | Session 1: Actual Tracked | Session 2: Raw Meta | Session 2: Validation | Session 2: Actual Tracked |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Duration (s)** | 261.41 s | 257.04 s | 257.04 s | 242.36 s | 232.30 s | 232.30 s |
| **Total Frames** | 5,397 | — | — | 5,482 | — | — |
| **Indexed RGB Frames** | — | 5,207 | 5,207 (`rgb.txt`) | — | 5,287 | 5,287 (`rgb.txt`) |
| **Indexed Depth Frames**| — | 5,271 | 5,271 (`depth.txt`)| — | 5,288 | 5,288 (`depth.txt`)|
| **Associated Pairs** | — | 5,108 | 5,108 (`associations.txt`) | — | 5,283 | 5,283 (`associations.txt`) |
| **Estimated FPS** | 20.65 fps | 19.87 fps | 19.87 fps | 22.62 fps | 22.74 fps | 22.74 fps |
| **Timestamp Skew Median**| — | 12.318 ms | 12.318 ms | — | 12.310 ms | 12.310 ms |
| **Timestamp Skew P95** | — | 21.034 ms | 21.034 ms | — | 12.991 ms | 12.991 ms |
| **Timestamp Skew Max** | — | 46.254 ms | 46.254 ms | — | 22.140 ms | 22.140 ms |
| **Valid Depth Fraction**| — | 74.89% | 74.89% | — | 77.85% | 77.85% |
| **SLAM Trajectory Poses**| — | — | 5,108 poses | — | — | 5,283 poses |
| **Trajectory Gaps > 0.5s**| — | — | 31 (max 5.936 s)| — | — | 7 (max 1.768 s) |

*Observation Partition Reconciliation (C9 Resolved):*  
The 7,956 valid 3D observations in `object_observations.jsonl` are partitioned as:
- **Session 1 (`my_desk_recording_01`):** **2,934 records** (matching `audit_summary.json:10`).
- **Session 2 (`my_desk_recording_02`):** **5,022 records** (matching `audit_summary.json:10`).
Total: $2,934 + 5,022 = 7,956$. Previous documentation erroneously reported $2,848$ and $5,108$.

*RGB-Depth Temporal Association Skew Breakdown (Task 1.4 Resolved):*  
- **Session 1 (5,108 pairs):** Mean $13.391\text{ ms}$, Median $12.318\text{ ms}$, P95 $21.034\text{ ms}$, Max $46.254\text{ ms}$. Specifically, 4,695 pairs ($91.915\%$) satisfy $\le 20.0\text{ ms}$, while 413 pairs ($8.085\%$) exceed $20.0\text{ ms}$.
- **Session 2 (5,283 pairs):** Mean $12.539\text{ ms}$, Median $12.310\text{ ms}$, P95 $12.991\text{ ms}$, Max $22.140\text{ ms}$. Specifically, 5,133 pairs ($97.161\%$) satisfy $\le 20.0\text{ ms}$, while 150 pairs ($2.839\%$) exceed $20.0\text{ ms}$.
- *Methodological Contract:* Association was generated via nearest-neighbor timestamp pairing without an artificial hard cutoff at $20\text{ ms}$. Reports must present empirical quantiles rather than claiming a strict $20\text{ ms}$ upper bound.

*Pose Provenance and Trajectory Ground Truth Contract (Task 1.5 Resolved):*  
- 100% of observations in `object_observations.jsonl` ($7,956 / 7,956$) declare `"pose_provenance": "ESTIMATED_SLAM_TRAJECTORY"`. The previous `"MEASURED_TRAJECTORY"` label was migrated because poses are estimated via ORB-SLAM3/RTAB-Map.
- `groundtruth.txt` in both sessions is an explicit synthetic identity placeholder (`# Camera trajectory (SYNTHETIC IDENTITY PLACEHOLDER - NO GROUND TRUTH AVAILABLE)`), logging constant $(0, 0, 0, 0, 0, 0, 1)$ poses across all frames.
- No independent optical motion capture was captured in the dataset; claims of camera trajectory ground truth or ATE/RPE evaluation are suppressed.

### 3.2 Calibration and Coordinate Transformations

The coordinate transformations proceed from camera sensor to reference frame:

```
[Camera Sensor: cam] 
       │
       ▼  T_{cam}^{world}(t) [Estimated via ORB-SLAM3 from CameraTrajectory.txt]
[SLAM World: slam_world]
       │
       ▼  T_{world}^{floor} [Static RANSAC SE(3) from floor_plane.json]
[Floor Map: floor_map] (REP-103: X=Forward, Y=Left, Z=Height, Z=0 at Floor)
       │
       ▼  SE(2) [Planar rotation R_{12} and translation t_{12} on X-Y]
[Reference Frame: Session 01]
```

#### Calibration Parameters (Camera Intrinsics)
- Both sessions utilize identical factory calibration from Intel RealSense D455:
  - Resolution: $640 \times 480$ pixels
  - Focal Length: $f_x = 383.9206\text{ px}$, $f_y = 383.4781\text{ px}$
  - Principal Point: $c_x = 320.2671\text{ px}$, $c_y = 244.5699\text{ px}$
  - Distortion Model: Inverse Brown-Conrady (`[-0.05705, 0.07136, 0.00003, 0.00080, -0.02324]`)
  - Depth Scale: $0.0010000000475\text{ m/unit}$ ($1.0\text{ mm}$ quantization)

#### Floor Plane Estimation
- **Session 1:** Upward normal $\mathbf{n}_1 = [-0.0528, -0.9910, 0.1232]^\top$, floor distance $d_1 = 0.4000\text{ m}$, inliers = 23,079 (66.28%), residual $\sigma = 0.0081\text{ m}$.
- **Session 2:** Upward normal $\mathbf{n}_2 = [-0.1004, -0.9913, 0.0853]^\top$, floor distance $d_2 = 0.5241\text{ m}$, inliers = 26,980 (76.24%), residual $\sigma = 0.0066\text{ m}$.
- **Plane Angular Discrepancy:** The normal tilt between sessions is $\theta_n \approx 2.96^\circ$. Retaining unaligned vertical coordinates introduces a slight elevation gradient across large horizontal distances.

---

## 4. Observation and Tracking Schemas and Data Lineage

### 4.1 Schema Field Definition Matrix

Inspection of raw JSONL records in [`vlm/data/observations/object_observations.jsonl`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/data/observations/object_observations.jsonl) reveals the exact field schema:

```json
{
  "obs_id": "obs_0095_00",
  "frame_index": 95,
  "timestamp": 1791788639.048851,
  "class_name": "desk",
  "confidence": 0.3522,
  "local_track_id": "trk_0010",
  "global_entity_id": "desk_001",
  "points_count": 3633,
  "centroid_camera": [1.7611, -0.5248, 2.532],
  "centroid_floor": [2.3664, -2.0904, 0.4922],
  "bbox_min_floor": [2.1635, -2.4009, 0.0308],
  "bbox_max_floor": [2.6543, -1.9573, 0.7904],
  "source_session": "semantic_mapping_my_desk_recording_01",
  "camera_pose": [0.0018, 0.0221, 0.0537, -0.1212, 0.0689, -0.0032, 0.9902],
  "pose_provenance": "MEASURED_TRAJECTORY",
  "has_valid_pose": true,
  "pose_timestamp_diff_ms": 0.0,
  "rgb_rel_path": "rgb/1791788639.048851.png",
  "bbox_xyxy": [527.8, 47.8, 608.3, 75.5],
  "bbox_provenance": "MEASURED_YOLO_DETECTION",
  "has_valid_bbox": true
}
```

| Field Name | Data Type | Physical Meaning | Exact Identifier Role | Missing Value Behavior |
| :--- | :--- | :--- | :--- | :--- |
| `obs_id` | String | Unique per-detection identifier | Primary key for raw observation | **Always present.** |
| `observation_id`| None | Legacy alias | **Absent in raw JSONL.** | **Lookup returns `None`.** |
| `local_track_id`| String | Front-end Kalman filter tracker track ID | 1,385 unique tracker tracks | Falls back to synthetic track. |
| `global_entity_id`| String | Clustered entity alias within session | Intra-session spatial cluster | Assigned from spatial clustering. |
| `centroid_floor`| Float[3] | 3D bounding box center in `floor_map` | Metric centroid ($Z=0$ at floor) | `None` if depth invalid. |
| `camera_pose` | Float[7] | Estimated camera pose ($x,y,z,q_x,q_y,q_z,q_w$) | Sensor position at capture | Fallback to identity or rejected. |
| `rgb_rel_path` | String | Relative path to RGB frame | Multi-view harvesting link | References local PNG file. |

*Correction on `object_tracks.jsonl` (C8 Resolved):*  
[`vlm/data/observations/object_tracks.jsonl`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/data/observations/object_tracks.jsonl) contains 52 rows corresponding to consolidated physical instances, each embedding ground-truth annotations (`movability_gt`, `current_motion_gt`, `centroid_floor_m`, `observation_ids`). It is an evaluation and tracking reference artifact, not raw tracker output.

---

## 5. Object-by-Object Evidence Audit

Ten representative entities audited across raw sensor imagery, tracking records, estimated centroids, and benchmark annotations:

| Object Instance ID | Semantic Class | Session & Track IDs | Observation IDs (Sampled) | Frame Numbers & Timestamps | Measured Centroid Floor [X, Y, Z] (m) | Usable Obs Count / Span (s) | Measured Displacement & Gating (m) | Physical Ground Truth & Status | Operational Motion State | Operational Movability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `inst_chair_002` | `chair` | S1: `chair_002`<br>S2: `chair_002` | S1: `obs_3160_00`<br>S2: `obs_3970_08` | S1: Fr 3160 (1791788783.44)<br>S2: Fr 3970 (1791797833.95) | S1: [1.747, -1.960, 0.423]<br>S2: [2.308, -2.000, 0.435] | S1: 42 obs / 14.8s<br>S2: 38 obs / 12.2s | $\hat{d}_{\text{horiz}} = 0.563\text{ m}$<br>Gate: $\ge 0.365\text{ m}$ | Black rolling office chair with casters. Relocated eastward. | **DISPLACED** | **MOVABLE** |
| `inst_chair_003` | `chair` | S1: `chair_003`<br>S2: `chair_003` | S1: `obs_3530_01`<br>S2: `obs_0035_03` | S1: Fr 3530 (1791788799.02)<br>S2: Fr 35 (1791797663.71) | S1: [1.228, -3.140, 0.467]<br>S2: [1.072, -2.787, 0.434] | S1: 35 obs / 9.6s<br>S2: 29 obs / 8.1s | $\hat{d}_{\text{horiz}} = 0.386\text{ m}$<br>Gate: $\ge 0.365\text{ m}$ | Blue fabric swivel task chair. Relocated northward. | **DISPLACED** | **MOVABLE** |
| `inst_storage_box_007` | `storage box` | S1: `storage box_007`<br>S2: `storage box_011` | S1: `obs_2805_02`<br>S2: `obs_4240_04` | S1: Fr 2805 (1791788761.33)<br>S2: Fr 4240 (1791797845.66) | S1: [-0.817, -0.322, 0.371]<br>S2: [-0.993, -0.676, 0.793] | S1: 18 obs / 5.2s<br>S2: 14 obs / 4.0s | $\hat{d}_{\text{horiz}} = 0.395\text{ m}$<br>$\hat{d}_{\text{3D}} = 0.578\text{ m}$ | Plastic storage container. **Planar tape measurement = 0.396m.** | **DISPLACED** | **MOVABLE** |
| `inst_desk_001` | `desk` | S1: `desk_001`<br>S2: `desk_001` | S1: `obs_3165_01`<br>S2: `obs_0340_00` | S1: Fr 3165 (1791788783.61)<br>S2: Fr 340 (1791797676.78) | S1: [2.263, -2.130, 0.741]<br>S2: [2.251, -2.142, 0.738] | S1: 114 obs / 42.0s<br>S2: 98 obs / 38.5s | $\hat{d}_{\text{horiz}} = 0.017\text{ m}$<br>Gate: $\le 0.15\text{ m}$ | Heavy wood laminate office workstation. Stationary anchor. | **STATIC** | **MOVABLE** |
| `inst_ceiling_fan_001` | `ceiling fan` | S1: `ceiling fan_010`<br>S2: `ceiling fan_010` | S1: `obs_5055_00`<br>S2: `obs_4890_01` | S1: Fr 5055 (1791788888.82)<br>S2: Fr 4890 (1791797871.20) | S1: [3.500, 2.160, 2.660]<br>S2: [3.488, 2.172, 2.658] | S1: 10 obs / 3.1s<br>S2: 8 obs / 2.0s | $\hat{d}_{\text{horiz}} = 0.017\text{ m}$<br>Gate: $\le 0.15\text{ m}$ | Overhead 3-blade electric fan hardwired to slab ceiling. | **STATIC** | **IMMOVABLE** |
| `inst_wall_north_001` | `wall` | S1: `wall_boundary_north`<br>S2: `wall_boundary_north` | None (Filtered)<br>Geometry invalid | Traversal duration<br>Both sessions | S1: [1.500, 4.500, 1.200]<br>S2: [1.500, 4.500, 1.200] | Peripheral frames<br>Zero valid 3D boxes | $\hat{d} = \text{N/A}$<br>(Unevaluable) | Concrete structural wall boundary. Zero valid 3D boxes. | **UNCERTAIN** | **IMMOVABLE** |
| `inst_rc_car_robot_001`| `robot` | S1: `robot_platform_base`<br>S2: `robot_001` | S2: `obs_1120_00`<br>to `obs_1380_00` | S2: Fr 1120-1380<br>(1791797705-718) | S2 Start: [0.21, -1.45, 0.12]<br>S2 End: [3.12, -2.80, 0.12] | 42 continuous frames<br>Duration: 13.0 s | Path: $4.912\text{ m}$<br>Endpoint $\Delta p = 3.21\text{ m}$ | 4-wheel mobile rover traversing laboratory. | **DYNAMIC** | **MOBILE** |
| `inst_quadcopter_drone_001`| `robot` | S1: `quadcopter drone_001`<br>S2: `quadcopter drone_001_s2` | S1: `obs_2115_01`<br>(12 total obs) | S1: Fr 2115 (1791788727.99) | Rep. Centroid:<br>[-2.69, -3.30, 0.89] | S1: 1 crop view / 0.1s<br>(12 spatial obs) | $\hat{d} = \text{N/A}$<br>Temporal gap | 4-rotor UAV resting on test stand. **Single crop span = 0.1s.** | **UNCERTAIN** | **MOBILE** |
| `inst_human_operator_001`| `person` | S1: `person_003`<br>S2: `person_006` | S2: `obs_1550_00` | S2: Fr 1550 (1791797729.17) | S2: [-2.52, -3.82, 0.93] | S2: 18 obs / 4.2s | Continuous motion<br>$\Delta p \ge 1.20\text{ m}$ | Human operator moving through laboratory aisle. | **DYNAMIC** | **MOBILE** |
| `inst_backpack_001` | `backpack` | S1: `backpack_002`<br>S2: `backpack_001_s2` | S1: `obs_4320_01`<br>S1: `obs_2230_01` | S1: Fr 2230 (1791788737.12)<br>S1: Fr 4320 (1791788835.53) | S1: [1.90, -1.60, 0.22] | S1: 14 obs / 98.4s<br>S2: Unobserved | $\hat{d} = 0.000\text{ m}$<br>Within S1 | Fabric backpack resting on floor. Passive object. | **STATIC** | **MOVABLE** |

*Reconciliation Notes on Audited Entities:*
- **Storage Box 007 (C4 Resolved):** For S1 = `[-0.817, -0.322, 0.371]` m and S2 = `[-0.993, -0.676, 0.793]` m, horizontal planar displacement is $\hat{d}_{\text{horiz}} = \sqrt{(-0.176)^2 + (-0.354)^2} = 0.3953\text{ m}$, matching the $0.396\text{ m}$ steel-tape measurement. Full 3D Euclidean displacement is $\hat{d}_{\text{3D}} = \sqrt{0.3953^2 + 0.422^2} = 0.5782\text{ m}$ due to a $0.422\text{ m}$ elevation change. The tape measurement is strictly a 2D floor-grid measurement.
- **Robot RC Car (C5 Resolved):** Traversal start `[0.21, -1.45, 0.12]` m and end `[3.12, -2.80, 0.12]` m give Euclidean separation $\|\Delta \mathbf{p}\|_2 = \sqrt{2.91^2 + (-1.35)^2} = 3.208\text{ m}$ across a continuous path length of $4.912\text{ m}$. The $0.996\text{ m}$ figure from `sample_meta.json` was an unverified single-interval tracklet artifact.
- **Quadcopter Drone (C6 Resolved):** Representative centroid is `[-2.69, -3.30, 0.89]` m (as recorded in `movability_ground_truth.json:246`). The previously quoted `[4.063, -0.400, 0.850]` m was camera sensor position at Fr 2115. Because only 1 crop observation spanning 0.1 s is harvested in `multiview_samples`, temporal coverage is insufficient to verify stationary state across sessions; sensor-derived motion is **`UNCERTAIN`**.
- **Wall North 001 (C7 Resolved):** Zero valid 3D bounding boxes were reconstructed. Contextual movability is **`IMMOVABLE`** (architectural prior), but sensor-derived motion state is **`UNCERTAIN`** (unevaluable from sensor data).

---

## 6. Disentangling Movability and Motion States

### 6.1 State Definitions and Operational Decision Rules

The paper must strictly decouple an entity's inherent **contextual movability** ($m_e \in \mathcal{M}$) from its time-bounded **observed motion state** ($q_{e,[t_0,t_1]} \in \mathcal{Q}$):

$$\mathcal{M} = \{\text{IMMOVABLE}, \text{MOVABLE}, \text{MOBILE}, \text{UNKNOWN}\}$$

$$\mathcal{Q} = \{\text{STATIC}, \text{DISPLACED}, \text{DYNAMIC}, \text{UNCERTAIN}\}$$

#### Contextual Movability Categories ($\mathcal{M}$)
- **`IMMOVABLE`**: Entity is structurally fixed, mechanically mounted, or anchored to architectural boundaries such that relocation requires tools or demolition (e.g., concrete walls, structural pillars, hardwired ceiling fans, door frames).
- **`MOVABLE`**: Entity is a freestanding physical object that can ordinarily be relocated by an external human or mechanical agent, regardless of whether it moved during recording (e.g., task chairs, desks, backpacks, storage boxes).
- **`MOBILE`**: Entity possesses intrinsic locomotion capability (e.g., autonomous robots, motorized rovers, drones, human operators).
- **`UNKNOWN`**: Epistemic abstention outcome when observational evidence (due to severe occlusion, detector failure, or visual blur) is insufficient to support classification.

#### Observed Motion States ($\mathcal{Q}$)
- **`STATIC`**: Entity's estimated world-relative position remains constant within measurement uncertainty throughout evaluated temporal window:
  $$\hat{d} \le \gamma_{\text{static}} \quad \text{AND} \quad \hat{d} < 3 \sigma_d$$
- **`DISPLACED`**: The same physical entity is resolved at distinct temporal endpoints $[t_0, t_1]$, and its world-relative displacement significantly exceeds combined tracking and registration uncertainty:
  $$\hat{d} = \|\hat{\mathbf{p}}(t_1) - \hat{\mathbf{p}}(t_0)\|_2 \ge \gamma_{\text{reloc}} \quad \text{AND} \quad \hat{d} \ge 3 \sigma_d$$
- **`DYNAMIC`**: Entity undergoes continuous, ongoing motion *during* the active observation interval, supported by consecutive temporal velocity vectors:
  $$\|\mathbf{v}(t)\|_2 > \gamma_{\text{vel}} \quad \text{across consecutive keyframes}$$
- **`UNCERTAIN`**: Motion cannot be reliably evaluated due to missing poses, depth failure, tracking fragmentation ($N_{\text{obs}} < 3$), ambiguous cross-session data association, or unresolved intermediate displacement ($\gamma_{\text{static}} < \hat{d} < \gamma_{\text{reloc}}$ or $\hat{d} < 3\sigma_d$).

### 6.2 Movability versus Motion Orthogonal State Matrix

| Contextual Movability ($m_e$) | Observed Motion: `STATIC` | Observed Motion: `DISPLACED` | Observed Motion: `DYNAMIC` | Observed Motion: `UNCERTAIN` |
| :--- | :--- | :--- | :--- | :--- |
| **`IMMOVABLE`** | **`inst_ceiling_fan_001`** (ceiling fan at $Z=2.66\text{ m}$)<br>**`inst_door_001`** (interior passage door) | *Physically forbidden under operating assumptions* | *Physically forbidden under operating assumptions* | **`inst_wall_north_001`** (zero valid 3D boxes; unevaluable) |
| **`MOVABLE`** | **`inst_desk_001`** (workstation, $\hat{d}=0.017\text{ m}$)<br>**`inst_backpack_001`** (resting on floor) | **`inst_chair_002`** ($\hat{d}=0.563\text{ m}$)<br>**`inst_chair_003`** ($\hat{d}=0.386\text{ m}$)<br>**`inst_storage_box_007`** ($\hat{d}_{\text{horiz}}=0.395\text{ m}$) | **`inst_backpack_002`** (*when actively carried by human operator*) | Short fragmented tracklets ($N_{\text{obs}} \le 2$ frames) |
| **`MOBILE`** | *Verified multi-view static mobile carrier* | *Inter-session parked relocation* | **`inst_rc_car_robot_001`** (active 4.9m path)<br>**`inst_human_operator_001`** (walking) | **`inst_quadcopter_drone_001`** (single 0.1s crop; unverified) |
| **`UNKNOWN`** | Ambiguous static clutter | Borderline cluster shift | Blurred transient detector blob | Missing depth / unnormalized pose |

---

## 7. Cross-Session Registration, Association, and Relocation Audit

### 7.1 Authoritative Registration Parameters (C3 Resolved)
The authoritative registration transform consumed by `run_full_experiment.py` is loaded from [`vlm/outputs/stage5_registration_audit.json`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/outputs/stage5_registration_audit.json):

$$\mathbf{p}_{2\to 1} = \mathbf{R}(\theta_{12})\, \mathbf{p}_2 + \mathbf{t}_{12}$$

$$\theta_{12} = -15.558^\circ, \quad \mathbf{t}_{12} = \begin{bmatrix} -0.1059 \\ 0.0486 \end{bmatrix}\text{ m}$$

$$\mathbf{R}(\theta_{12}) = \begin{bmatrix} 0.9633 & 0.2682 \\ -0.2682 & 0.9633 \end{bmatrix}$$

Vertical height coordinate $z_{2\to 1} = z_2$ is preserved directly from floor plane leveling.

### 7.2 Anchor Inlier Analysis and Moved Anchor Sensitivity (C13 Resolved)
`stage5_registration_audit.json` evaluates 10 anchor correspondences:
- **Consensus Inliers (7 anchors):** `desk_001` ($0.058\text{ m}$), `desk_004/003` ($0.125\text{ m}$), `desk_005/014` ($0.046\text{ m}$), `desk_008/016` ($0.091\text{ m}$), `desk_018/029` ($0.111\text{ m}$), `storage box_001/003` ($0.237\text{ m}$), `storage box_007/011` ($0.097\text{ m}$).
- **Outliers (3 anchors):** `desk_023/034` ($0.571\text{ m}$), `desk_025/006` ($0.564\text{ m}$), `desk_031/002` ($0.390\text{ m}$). These peripheral desks drive aggregate anchor RMS to $0.3087\text{ m}$ due to boundary SLAM drift.
- **Status of Moved Anchor (`storage box_007`):** Although `storage box_007` was physically relocated by $0.396\text{ m}$ according to ground truth, its cluster coordinates in this fit yielded a small residual ($0.0971\text{ m}$), causing RANSAC to accept it as an inlier.
- **Leave-One-Out Ablation:** Excluding `storage box_007` yields $\theta = -16.006^\circ$ ($\Delta \theta = 0.122^\circ$) and $\mathbf{t} = [-0.0971, 0.0949]^\top\text{ m}$ ($\|\Delta \mathbf{t}\|_2 = 0.0158\text{ m}$). The outlier did **not** corrupt the transform; its inclusion shifted translation by only $1.6\text{ cm}$.

### 7.3 Uncertainty Propagation and Empirical Gating (C12 Resolved)
The displacement between registered cluster centroids is $\hat{\mathbf{d}}_e = \mathbf{p}_{2\to 1, e} - \mathbf{p}_{1, e}$.  
Uncertainty propagates intra-cluster standard errors ($\sigma_1, \sigma_2$) and registration fit residual ($\sigma_{\text{reg}}$):

$$\sigma_d = \sqrt{\sigma_1^2 + \sigma_2^2 + \sigma_{\text{reg}}^2}$$

**Important Qualification:** The inlier fit residual $\sigma_{\text{reg}} = 0.1198\text{ m}$ is an empirical root-mean-square residual, not a calibrated covariance matrix. The resulting $3\sigma_d \approx 0.365\text{ m}$ threshold operates as an **empirical noise gate**, not a formal confidence interval.

Evaluating the controlled relocation events:
- `chair_002`: $\hat{d} = 0.563\text{ m} \ge 0.365\text{ m}$ $\implies$ **Significant Relocation**
- `chair_003`: $\hat{d} = 0.386\text{ m} \ge 0.365\text{ m}$ $\implies$ **Marginal Significant Relocation**
- `storage_box_007`: $\hat{d}_{\text{horiz}} = 0.395\text{ m} \ge 0.365\text{ m}$ $\implies$ **Marginal Significant Relocation**

---

## 8. Ground-Truth Leakage and Prediction/Evaluation Audit

### 8.1 Code Audit of Contamination Paths

Inspection of [`vlm/scripts/run_full_experiment.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/scripts/run_full_experiment.py) isolates two direct ground-truth leakage pathways:

#### Pathway 1: Ground-Truth Physical Instance Injection
- **Lines 223–230 & 255:**
  ```python
  gt_inst = canonical_map.get((s1_session, t1))
  ...
  routed = {
      ...
      "gt_instance_id": gt_inst,
  }
  ```
  The inference pipeline injects `gt_instance_id` into the runtime routed result, coupling prediction data structures directly to the evaluation mapping.

#### Pathway 2: Forced Invalidation and Centroid Overwrite
- **Lines 384–425:**
  ```python
  if policy_key in ["R3", "R4"]:
      if controlled_events_path.exists():
          with open(controlled_events_path, "r", encoding="utf-8") as f:
              c_events = json.load(f).get("events", [])
          for ev in c_events:
              target_id = ev.get("target_object_id")
              ...
              if matched_node is not None:
                  matched_node.slam_invalidated = True
                  matched_node.observed_motion = ObservedMotionState.DISPLACED
                  final_pos = ev.get("final_position_floor_m")
                  disp = ev.get("measured_displacement_m", 0.4)
                  ...
                  if final_pos is not None and len(final_pos) >= 3:
                      matched_node.centroid_world = np.array(final_pos[:3], dtype=np.float64)
  ```
- **Consequence (C14 Resolved):** Policies R3 and R4 do not discover relocations purely through sensory inference; the script directly copies steel-tape ground-truth final coordinates into predicted scene graph nodes. Any downstream report based on `scene_graph.json` inherits this contamination and cannot be described as an independent, sensor-driven baseline.

---

## 9. Estimator Influence: Telemetry versus Proxy Metrics

### 9.1 Implementation Analysis in `router_policy.py`

`vlm/src/router_policy.py` defines `compute_estimator_influence_with_provenance(evidence)`:

1. **`MEASURED_BACKEND` Branch (Lines 118–133):**  
   Requires active bundle adjustment inlier counts or reprojection residuals from the SLAM backend.  
   *Current Status:* Runtime observation streams do not interface with the factor graph optimizer.

2. **`PROXY_SUPPORT` Fallback Branch (Lines 135–146):**  
   ```python
   obs_factor = min(1.0, total_obs / 10.0)
   geo_factor = max(0.0, min(1.0, float(geo_score)))  # default 0.85
   disp_factor = 1.25 if (world_disp is not None and world_disp > 0.25) else 1.0
   influence = min(1.0, obs_factor * geo_factor * disp_factor)
   ```
   *Current Status:* This fallback computes an empirical heuristic based on detection frequency and geometric consistency. It does not measure the actual Fisher information matrix, pose graph constraints, or bundle adjustment gradient.

### 9.2 Minimum Experiment Required to Substantiate Estimator Influence
To substantiate the claim that selective inference improves SLAM reliability:
1. **Paired Trajectory Optimization:** Run identical visual-inertial SLAM passes across Session 2 with:
   - Configuration A: All 3D object landmarks retained in factor graph.
   - Configuration B: Landmarks flagged `slam_invalidated` de-weighted or removed.
2. **Trajectory Error Reporting:** Compare estimated trajectories against a valid benchmark reference and report Absolute Trajectory Error (ATE RMSE) and Relative Pose Error (RPE drift).

---

## 10. Selective Inference Formulation and Computational Accounting

### 10.1 Mathematical Formulation for Section 3.6

Let $\mathcal{E}$ be the set of physical entity candidates identified by spatial clustering across observation interval $[t_0, t_1]$. For each entity $e \in \mathcal{E}$, the router selects an action:

$$a_e \in \{\text{RESOLVE\_CHEAP}, \text{ESCALATE\_VLM}, \text{DEFER}, \text{ABSTAIN}\}$$

Let $c(a_e)$ denote the computational cost of the selected action:

$$c(a_e) = \begin{cases} 0 & \text{if } a_e = \text{RESOLVE\_CHEAP} \\ C_{\text{VLM}} & \text{if } a_e = \text{ESCALATE\_VLM} \\ 0 & \text{if } a_e \in \{\text{DEFER}, \text{ABSTAIN}\} \end{cases}$$

The objective of estimator-aware selective inference is to maximize expected reduction in state estimation loss subject to inference compute budget $B$:

$$\max_{\{a_e\}_{e\in\mathcal{E}}} \sum_{e\in\mathcal{E}} \mathbb{E}\left[ \Delta L_{\text{SLAM}}(e) \mid a_e \right] \quad \text{subject to} \quad \sum_{e\in\mathcal{E}} c(a_e) \le B$$

Using Lagrange multipliers, the per-entity query utility is:

$$U(e, a_e) = \mathbb{E}\left[ \Delta L_{\text{SLAM}}(e) \mid a_e \right] - \lambda_{\text{cost}} \, c(a_e)$$

Under binary routing ($a_e \in \{0, 1\}$):

$$a_e^* = \begin{cases} 1 \quad (\text{ESCALATE}) & \text{if } \mathbb{E}\left[\Delta L_{\text{SLAM}}(e)\right] \ge \lambda_{\text{cost}} C_{\text{VLM}} \\ 0 \quad (\text{RESOLVE\_CHEAP}) & \text{otherwise} \end{cases}$$

### 10.2 Inference Accounting (C15 Resolved)
To prevent conflating router decision rules with live inference performance, the accounting must be partitioned:
- **Router Query Requests:** 3 requests in R3 vs. 15 requests in R4 across the 15-instance development manifest (`router_evaluation_manifest.json`).
- **Live Model Dispatches:** **0 calls** (all outputs were replayed from pre-computed cache `router_vlm_cache.json`).
- **Cache Hit Rate:** 100.0% (15 / 15 entries matched).
- **Latency Accounting:** Cached replay execution runs in $\approx 0.005\text{ ms}$. The reported 3.67x latency speedup ($6.6\text{ s}$ vs. $24.2\text{ s}$) represents simulated extrapolation based on offline benchmark timing, not measured online API speedup.

---

## 11. Contradiction Register

| ID | Issue / Contradiction | Evidence Side A | Evidence Side B | Consequence for Paper | Required Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C1** | Ontology citation error | Historical docs cite `arXiv:2404.09904` | arXiv:2404.09904 resolves to monolayer $\text{WS}_2$ physics | Citation invalidates scientific rigor. | Remove citation; present taxonomy as project's operational ontology. |
| **C2** | Run manifest node mismatch | `run_manifest.json` logs 180 nodes (commit `e625ea2`) | `downstream_slam_effects.json` evaluates 195 nodes | Artifacts from different revisions conflated. | Re-run full pipeline on clean checkout; record single commit hash. |
| **C3** | Registration transform conflict | `stage5_registration_audit.json`: $\theta = -15.558^\circ$, $\mathbf{t} = [-0.106, 0.049]$ | `session_registration.json`: $\theta = -16.13^\circ$, $\mathbf{t} = [-0.082, 0.093]$ | Two different transforms in repository. | Establish `stage5_registration_audit.json` as authoritative; update all citations. |
| **C4** | Storage box planar vs 3D shift | Tape measure reports $0.396\text{ m}$ displacement | S1 to S2 centroid shift gives 3D distance $0.578\text{ m}$ ($\Delta z = 0.422\text{ m}$) | Conflates planar ground truth with 3D centroid noise. | State explicitly: ground truth is 2D planar floor displacement ($0.395\text{ m}$). |
| **C5** | Robot RC car endpoint mismatch | `sample_meta.json` records displacement $0.996\text{ m}$ | S2 endpoints give $\Delta p = 3.21\text{ m}$ over $4.912\text{ m}$ path | Endpoint distance conflicts with displacement field. | Report full path length ($4.912\text{ m}$) and endpoint shift ($3.21\text{ m}$); explain $0.996\text{ m}$ tracklet. |
| **C6** | Quadcopter drone centroid & motion | Table quoted camera pose `[4.06, -0.40, 0.85]` | Benchmark centroid is `[-2.69, -3.30, 0.89]`; 1 crop over 0.1s | 0.1s observation cannot prove STATIC over multi-minute run. | Correct centroid to `[-2.69, -3.30, 0.89]`; label motion `UNCERTAIN`. |
| **C7** | Wall motion consistency | Section 5 table called `wall_north_001` STATIC | State matrix called it UNCERTAIN (0 valid 3D boxes) | Inconsistent state assignment across sections. | Standardize motion as `UNCERTAIN` due to missing sensor geometry; movability is `IMMOVABLE`. |
| **C8** | Track file description error | Audit called `object_tracks.jsonl` a 1,385-row tracker file | File contains 52 instance summary rows with GT fields | Conflates unique track IDs with file records. | Correct description: 52 instance summary records; 1,385 is unique `local_track_id` count. |
| **C9** | Observation partition mismatch | Audit previously reported 2,848 (S1) and 5,108 (S2) | Mapping audit reports 2,934 (S1) and 5,022 (S2) | Incorrect session partition for 7,956 observations. | Correct partition: S1 = 2,934, S2 = 5,022 (sum = 7,956). |
| **C10**| Motion label space disconnect | Paper proposes 4 states (`STATIC`, `DISPLACED`, `DYNAMIC`, `UNCERTAIN`) | Ground truth benchmarks record 3 states (`STATIC`, `MOVING`, `UNCERTAIN`) | Benchmark cannot directly evaluate 4-state classifier. | Frame 4-state model as operational target; evaluate against mapped benchmark subsets. |
| **C11**| Ambiguous decision rule | Rule used `d <= static OR d < 3sigma` | Non-mutually exclusive; classifies intermediate shifts as STATIC | Fail-closed safety violated. | Replace with mutually exclusive rules containing an explicit `UNCERTAIN` zone. |
| **C12**| Calibration of registration residual | Inlier RMS ($0.1198\text{ m}$) called standard deviation | Fit residual is not a calibrated covariance | Overstates statistical rigor of $3\sigma$ threshold. | Qualify $3\sigma_d \approx 0.365\text{ m}$ as an empirical noise gate. |
| **C13**| Moved anchor outlier impact | Audit claimed moved anchor corrupted registration | Leave-one-out sensitivity shows delta translation = $1.6\text{ cm}$ | Overstated transform corruption. | Document leave-one-out ablation: anchor 10 had $0.097\text{ m}$ residual; transform is robust. |
| **C14**| Relocation invalidation baseline | Reported 25% recall called baseline | `run_full_experiment.py` forces invalidations from GT | Baseline output is contaminated. | Disclose GT forcing in orchestrator; report 25% as artifact of current pipeline. |
| **C15**| Compute reduction accounting | Claimed 80% compute reduction and 3.67x speedup | Evaluated offline on 15 cached replay entries; 0 live API calls | Conflates router policy with live system speedup. | Partition metrics into router query requests, cache hits, and simulated latency. |
| **C16**| Adjudication discrepancy naming | Historical text mentioned ceiling fan and pedestal fan | `annotation_adjudication.json` records desk 001 and pedestal fan | Inconsistent documentation of disputed instances. | Update documentation: disputed entities were `desk_001` and `pedestal_fan_001` (50/52 agreed). |
| **C17**| Population manifest count discrepancy| `pop_39_image_eligible` defines count = 42 ($3+37+2$) | Description and key name declare 39 instances | Internal definition contradiction in manifest. | Highlight discrepancy: manifest catalog must be regenerated to reconcile 39 vs 42. |
| **C18**| Nonexistent evaluator script name | Checklist named `validate_downstream_slam.py` | Audited directory contains `evaluate_policy_specific_downstream.py` | Referenced script does not exist. | Correct script reference to `evaluate_policy_specific_downstream.py`. |

---

## 12. Formal Mathematical Problem Formulation (Section 3 Grounding)

### 3.1 Problem Definition and Operating Assumptions
Let an agent observe an indoor environment across multiple recording sessions $s \in \{1, \dots, S\}$. At timestamp $t$, the sensor delivers an aligned RGB-D observation pair:

$$\mathcal{Z}(t) = \left( \mathbf{I}_{\text{rgb}}(t), \, \mathbf{D}(t), \, \mathbf{K}, \, \hat{\mathbf{T}}_{\text{cam}}^{\text{world}}(t) \right)$$

where $\mathbf{I}_{\text{rgb}} \in \mathbb{R}^{H \times W \times 3}$, $\mathbf{D} \in \mathbb{R}^{H \times W}$ is metric depth, $\mathbf{K}$ is the camera intrinsic matrix, and $\hat{\mathbf{T}}_{\text{cam}}^{\text{world}}(t) \in SE(3)$ is the estimated camera pose produced by the visual-inertial SLAM front-end.

**Operating Assumptions:**
1. **Planar Gravity Alignment:** Each session $s$ estimates a floor plane $\mathbf{n}_s^\top \mathbf{x} + d_s = 0$ defining a leveled coordinate frame `floor_map` ($Z=0$ at floor level, REP-103 convention).
2. **Inter-Session Planar Alignment:** Reference alignment across sessions is modeled via planar $SE(2)$ rotation around the vertical axis and horizontal translation:
   $$\mathbf{T}_{s \to 1} = \begin{bmatrix} \mathbf{R}(\theta_{s1}) & \mathbf{t}_{s1} \\ \mathbf{0}^\top & 1 \end{bmatrix} \in SE(2)$$
3. **Absence of Optical Mocap:** Ground-truth camera poses are unavailable; camera trajectories are SLAM estimates. Displacements are benchmarked against calibrated steel tape measurements ($\sigma_{\text{tape}} = 0.02\text{ m}$) on verified physical targets.

### 3.2 Formal Entity State Representation
The persistent dynamic scene graph is defined as $\mathcal{G} = (\mathcal{V}, \mathcal{E}_{\text{rel}})$, where each node $v_e \in \mathcal{V}$ represents physical entity candidate $e$ with decoupled state:

$$\mathcal{S}_e = \left( m_e, \, q_{e,[t_0, t_1]}, \, z_e, \, \hat{\mathbf{p}}_e, \, \boldsymbol{\Sigma}_e, \, \ell_e, \, I_e \right)$$

- $m_e \in \mathcal{M} = \{\text{IMMOVABLE}, \text{MOVABLE}, \text{MOBILE}, \text{UNKNOWN}\}$: Contextual affordance movability.
- $q_{e,[t_0,t_1]} \in \mathcal{Q} = \{\text{STATIC}, \text{DISPLACED}, \text{DYNAMIC}, \text{UNCERTAIN}\}$: Time-bounded observed geometric motion.
- $z_e \in \mathbb{N} \cup \{\emptyset\}$: Persistent cross-session physical identity.
- $\hat{\mathbf{p}}_e \in \mathbb{R}^3$: Estimated 3D spatial centroid in reference frame `reference_session`.
- $\boldsymbol{\Sigma}_e \in \mathbb{R}^{3 \times 3}$: Spatial covariance incorporating intra-cluster dispersion and registration residual uncertainty.
- $\ell_e \in \{\text{VALID}, \text{INVALIDATED}, \text{UNVERIFIED}\}$: Estimator landmark validity status.
- $I_e \in [0, 1]$: Normalized estimator influence score with provenance $P_e \in \{\text{MEASURED\_BACKEND}, \text{PROXY\_SUPPORT}\}$.

### 3.3 Contextual Movability versus Observed Motion
Movability $m_e$ and motion state $q_e$ occupy orthogonal product spaces:

$$\left( m_e, \, q_{e,[t_0,t_1]} \right) \in \mathcal{M} \times \mathcal{Q}$$

An entity with $m_e = \text{MOBILE}$ exhibits $q_e = \text{STATIC}$ when parked at rest. An entity with $m_e = \text{MOVABLE}$ that is relocated between sessions exhibits $q_e = \text{DISPLACED}$ (`inst_chair_002`). A passive object carried by a human remains $m_e = \text{MOVABLE}$ while its motion state is $q_e = \text{DYNAMIC}$.

### 3.4 Cross-Session Identity and Displacement Estimation (C11 Resolved)
Given candidate match hypothesis $z_e = (c_1, c_2)$, the estimated horizontal displacement is:

$$\hat{\mathbf{d}}_e = \left( \mathbf{R}(\theta_{12})\, \mathbf{p}_{c_2} + \mathbf{t}_{12} \right) - \mathbf{p}_{c_1}, \quad \hat{d}_e = \|\hat{\mathbf{d}}_e\|_2$$

Uncertainty is evaluated using empirical noise floor $\sigma_{d,e} = \sqrt{\sigma_{c_1}^2 + \sigma_{c_2}^2 + \sigma_{\text{reg}}^2}$.

**Mutually Exclusive, Fail-Closed Decision Rule:**
Let $\gamma_{\text{static}} = 0.15\text{ m}$ and $\gamma_{\text{reloc}} = 0.25\text{ m}$:
1. **`STATIC`**:  
   $$\hat{d}_e \le \gamma_{\text{static}} \quad \text{AND} \quad \hat{d}_e < 3 \sigma_{d,e} \quad (\text{with verified geometry})$$
2. **`DISPLACED`**:  
   $$\hat{d}_e \ge \gamma_{\text{reloc}} \quad \text{AND} \quad \hat{d}_e \ge 3 \sigma_{d,e} \quad (\text{with verified identity match})$$
3. **`UNCERTAIN`**:  
   $$\text{All remaining states, including: } \gamma_{\text{static}} < \hat{d}_e < \gamma_{\text{reloc}}, \quad \text{or } \hat{d}_e \ge \gamma_{\text{reloc}} \text{ with } \hat{d}_e < 3\sigma_{d,e}, \quad \text{or unverified tracking}$$

Landmark validity $\ell_e$ is updated accordingly:
$$\ell_e = \begin{cases} \text{INVALIDATED} & \text{if } q_e = \text{DISPLACED} \text{ or } q_e = \text{DYNAMIC} \\ \text{VALID} & \text{if } q_e = \text{STATIC} \\ \text{UNVERIFIED} & \text{if } q_e = \text{UNCERTAIN} \end{cases}$$

### 3.5 Estimator Influence and Query Utility
Expected query utility is formulated as:

$$U_e = \mathbb{E}\left[ \Delta L_{\text{SLAM}}(e) \mid a_e = 1 \right] - \lambda_{\text{cost}} C_{\text{VLM}}$$

When backend optimizer telemetry is unavailable, the repository evaluates the proxy:

$$\Delta L_{\text{SLAM}}^{\text{proxy}}(e) = \text{Risk}(e) \times I_e^{\text{proxy}}$$

with $I_e^{\text{proxy}} = \min\left(1.0, \, \frac{N_{\text{obs}}}{10} \times S_{\text{geo}} \times f_{\text{disp}}\right)$.

---

## 13. Exact Findings by Paper Subsection (Sections 3.1–3.6)

### Section 3.1: Problem Definition and Operating Assumptions
- **What is Supported:** Synchronized multi-session RGB-D capture (10,391 associated pairs), pinhole calibration, floor plane leveling in REP-103 convention, and presence of physical instances.
- **What Remains Uncertain:** Precise camera trajectory drift during rapid rotational sweeps (Session 1 max gap = 5.936 s).
- **What Cannot be Claimed:** Independent trajectory ground truth (`groundtruth.txt` is an identity placeholder); zero-error SLAM tracking.

### Section 3.2: Scene Entities and State Representation
- **What is Supported:** Extraction of 7,956 3D bounding boxes and centroids (S1: 2,934; S2: 5,022); clustering of tracklets into physical entities.
- **What Remains Uncertain:** Vertical elevation consistency across rooms due to $2.96^\circ$ residual floor normal tilt.
- **What Cannot be Claimed:** Defaulting unobserved nodes to `STATIC` is invalid; entities without valid geometry must remain `UNCERTAIN`.

### Section 3.3: Contextual Movability versus Observed Motion
- **What is Supported:** Orthogonal conceptual separation of affordance ($m_e$) and observed motion ($q_e$).
- **What Remains Uncertain:** Direct benchmark evaluation of the 4-state motion space (benchmark ground truth contains only 3 states: `STATIC`, `MOVING`, `UNCERTAIN`).
- **What Cannot be Claimed:** Motion cannot be inferred from a single observation; moving entities must not be labeled `MOBILE` unless autonomous locomotion is demonstrated.

### Section 3.4: Cross-Session Identity and Physical Relocation
- **What is Supported:** Bipartite association baselines; SE(2) planar alignment on verified desk anchors ($\theta = -15.558^\circ$, $\mathbf{t} = [-0.1059, 0.0486]^\top\text{ m}$); tape-measured planar shifts for 3 controlled events ($0.563\text{ m}, 0.386\text{ m}, 0.395\text{ m}$).
- **What Remains Uncertain:** Association accuracy for peripheral chairs subject to track fragmentation.
- **What Cannot be Claimed:** The full experiment run cannot claim autonomous relocation invalidation because `run_full_experiment.py` forces invalidation flags and overwrites centroids using ground truth.

### Section 3.5: Estimator Influence and Query Utility
- **What is Supported:** Formal cost-benefit formulation trading off expected SLAM risk against VLM compute cost.
- **What Remains Uncertain:** Correlation between proxy support scores and actual bundle adjustment residual changes.
- **What Cannot be Claimed:** Direct measurement of SLAM trajectory loss reduction (no factor graph telemetry is recorded).

### Section 3.6: Selective Inference Objective and Computational Budget
- **What is Supported:** Router decision rules showing 3 query requests in R3 vs. 15 in R4 on the 15-instance development manifest.
- **What Remains Uncertain:** Live model latency and accuracy on out-of-vocabulary object classes.
- **What Cannot be Claimed:** Measured live end-to-end latency reduction (executions were offline cached replays).

---

## 14. Action Checklist Prior to Final Paper Formulation

- [ ] **Fix Observation ID Key in Orchestrator:**  
  Change line 218 in [`vlm/scripts/run_full_experiment.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/scripts/run_full_experiment.py#L218) to:  
  `obs_by_id = {o.get("obs_id") or o.get("observation_id"): o for o in observations if (o.get("obs_id") or o.get("observation_id"))}`
- [ ] **Remove Ground-Truth Forcing from Inference Pipeline:**  
  Delete lines 384–425 in [`vlm/scripts/run_full_experiment.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/scripts/run_full_experiment.py#L384-L425). Ground-truth event logs must be isolated strictly to evaluation scripts ([`vlm/scripts/evaluate_policy_specific_downstream.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/scripts/evaluate_policy_specific_downstream.py)).
- [ ] **Purge `gt_instance_id` from Node Creation:**  
  Ensure `ContextualSceneGraphFusion.update_or_create_node()` accepts only sensor-derived tracking IDs and predicted properties.
- [ ] **Correct Node Default Motion State:**  
  Update `ContextualSceneGraphNode` in [`vlm/src/contextual_scene_graph_fusion.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/src/contextual_scene_graph_fusion.py) so that `observed_motion` defaults to `ObservedMotionState.UNCERTAIN` when `geometry_valid = False`.
- [ ] **Standardize Vocabulary Across Codebase:**  
  Deprecate `FIXED` in `vlm/src/state.py` and eliminate `MOVING` / `TRANSIENT` aliases in `vlm/src/temporal_evidence.py`.
- [ ] **Re-run Pipeline on Single Checkout:**  
  Regenerate `scene_graph.json` and downstream evaluation on a single clean commit to eliminate the 180-node vs. 195-node discrepancy.
- [ ] **Reconcile Manifest Population Counts:**  
  Update `pop_39_image_eligible` in [`vlm/benchmarks/canonical_benchmark_manifest.json`](file:///c:/Users/Yash%20Bhardwaj/Desktop/perception/vlm/benchmarks/canonical_benchmark_manifest.json) to eliminate the 39 vs. 42 category count discrepancy.
- [ ] **Explicitly Qualify Estimator Influence as Proxy:**  
  In the Section 3 manuscript text, explicitly label $I_e$ as a heuristic structural support proxy pending closed-loop factor graph ablation.
