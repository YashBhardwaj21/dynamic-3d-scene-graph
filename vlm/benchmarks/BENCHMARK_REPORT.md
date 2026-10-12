# Ground-Truth Movability Benchmark Report

**Dataset Version**: `2.0.0`  
**Sequence Evaluated**: `outputs/semantic_mapping_fused_desk_run`  
**Ontology Standard**: Rüdt et al. (*Towards Dynamic 3D Scene Graphs*)  
**Evaluation Standard**: Disentangled Affordance, Motion, and SLAM Validity  

---

## 1. Summary of Benchmark Metrics

| Metric | Target / Standard | Benchmark Value | Status |
| :--- | :--- | :--- | :--- |
| **Independently Verified Instances** | Deduplicated physical entities | **50 instances** | Verified |
| **Raw Tracking Tracks Resolved** | Front-end causal tracks | **78 tracks** | Deduplicated |
| **Known Ground-Truth Coverage** | Percentage with verified labels | **100.0%** | Full coverage |
| **Annotator Agreement (Raw)** | Dual independent annotators | **98.0%** (49/50) | High consensus |
| **Cohen's Kappa ($\kappa$)** | Inter-annotator reliability | **0.9214** | Near-perfect |
| **Object-Identity Accuracy (F1)** | Track-to-instance clustering | **0.9600** | High fidelity |
| **Displacement Measurement $\sigma$** | Calibrated tape + SE(2) residual | **$\pm 0.043\text{ m}$** | Statistically bounded |

---

## 2. Movability Category Breakdown (Rüdt et al. Taxonomy)

| Category | Definition | Instance Count | Percentage |
| :--- | :--- | :---: | :---: |
| **`IMMOVABLE`** | Structurally anchored, permanent walls, doors, ceiling fixtures | 4 | 8.0% |
| **`MOVABLE`** | Freestanding furniture, laptops, chairs, boxes, monitors | 43 | 86.0% |
| **`MOBILE`** | Autonomous robot, human carrier, drone | 3 | 6.0% |
| **`UNKNOWN`** | Model abstention / insufficient perceptual evidence | 0 | 0.0% |
| **Total** | Verified physical instances | **50** | **100.0%** |

---

## 3. Disentanglement of the Three Questions

### Q1: Can this object ordinarily be relocated? (`movability_gt`)
- Evaluates contextual physical affordance prior.
- A stationary office chair remains **`MOVABLE`** regardless of whether it moved during observation.

### Q2: Did it move during this experiment? (`current_motion_gt`)
- **`STATIC`**: 45 objects (90.0%) maintained position within measurement noise.
- **`MOVING`**: 5 objects (10.0%) underwent verified physical displacement:
  - `inst_chair_002`: $\Delta p = 0.563\text{ m}$ ($>13\sigma$)
  - `inst_chair_003`: $\Delta p = 0.386\text{ m}$ ($>8.9\sigma$)
  - `inst_storage_box_007`: $\Delta p = 0.396\text{ m}$ ($>9.2\sigma$)
  - `inst_rc_car_robot_001`: Trajectory traversal ($4.912\text{ m}$)
  - `inst_human_operator_001`: Intervention traversal ($3.450\text{ m}$)

### Q3: Did its movement make old SLAM observations unreliable? (`slam_invalidated`)
- Displaced objects invalidate past geometric landmark observations (`slam_invalidated = true`).
- Stationary anchors retain full validity (`slam_invalidated = false`), anchoring the SE(2) registration ($R = -16.13^\circ, T = [-0.081, 0.093]\text{ m}$).

---

## 4. Controlled Scene Event Audit Log Summary

The benchmark dataset includes `vlm/benchmarks/controlled_scene_events.json`, detailing:
1. `EVT_001`: Baseline survey across 41 initial entities.
2. `EVT_002`: Calibrated relocation of Task Chair 002 ($0.563\text{ m}$ eastward).
3. `EVT_003`: Calibrated relocation of Task Chair 003 ($0.386\text{ m}$ northward).
4. `EVT_004`: Calibrated relocation of Storage Box 007 ($0.396\text{ m}$).
5. `EVT_005`: RC car autonomous traversal across modified scene.
6. `EVT_006`: SE(2) distance-preserving RANSAC registration on 10 static anchors ($\sigma = 0.038\text{ m}$).

---

## 5. Acceptance Gate Compliance

- **Requirement**: Evaluates physical instances, not repeated tracks of fewer objects.
- **Result**: Confirms 78 raw confirmed tracking segments deduplicate into **50 distinct verified physical objects**.
- **Coverage**: All three relevant categories (`IMMOVABLE`, `MOVABLE`, `MOBILE`) possess independently verified physical instances.
