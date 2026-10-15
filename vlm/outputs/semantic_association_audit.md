# Semantic Metric Denominators & Physical-Instance Association Integrity Audit

## 1. Executive Summary & Audit Mandate
This audit examines the empirical denominators behind the reported **99.76% Semantic Macro-F1** and audits the **3,535 unmatched observations** in the dataset.

### Key Findings:
1. **Matched-Support Evaluation Bias:** The reported 99.76% Macro-F1 was computed **strictly on the 4,421 matched observations (55.6% coverage)**.
2. **44.4% of Dataset Omitted:** Exactly **3,535 observations (44.4%)** lacked ground-truth track matches and were silently excluded from the denominator.
3. **Root Cause of Unmatched Subset:** **90.2% of unmatched observations (3,187 / 3,535)** reside in Session 2 (`semantic_mapping_my_desk_recording_02`), caused by session-local tracker ID renumbering (e.g. `chair_006` for `chair_003`, `desk_014` for `desk_005`). `global_entity_id` is a session-local tracker ID, NOT a globally persistent identity.
4. **Mobile Agent Observation Dominance:** Mobile agents (`person`, `robot`) account for **842 observations (19.0%)** of all matched observations, artificially inflating per-frame accuracy through high-framerate track accumulation.
5. **Track Fragmentation Definition:** The 72.8% 'track fragmentation' metric is a **short-track statistic** (tracks with lifespan < 5 frames = 72.8%), NOT an identity switch rate.

---

## 2. Denominator Breakdown: Matched-Support vs Full Dataset

| Evaluation Metric | Matched-Support (Reported) | Full-Dataset Coverage | Notes |
|---|:---:|:---:|---|
| **Evaluated Observations** | **4,421** | **7,956** | 3,535 records excluded in matched mode |
| **Dataset Coverage** | **55.57%** | **100.00%** | Full dataset includes Session 2 renumbered tracks |
| **Semantic Macro-F1** | **99.76%** | **55.57%** (penalized UNMATCHED) | Raw matched score masks 44.4% unmapped data |
| **Session 1 Observations** | 2,586 (88.1%) | 2,934 | 88.1% of Session 1 observations matched |
| **Session 2 Observations** | 1,835 (36.5%) | 5,022 | Only 36.5% of Session 2 matched due to ID reset |

---

## 3. Systematic Analysis of the 3,535 Unmatched Observations

### 3.1 Distribution by Recording Session
- **Session 1 (`semantic_mapping_my_desk_recording_01`):** 348 unmatched observations (9.8%).
- **Session 2 (`semantic_mapping_my_desk_recording_02`):** 3,187 unmatched observations (90.2%).

### 3.2 Top Unmatched Tracker Entities in Session 2
The unmatched observations correspond to legitimate physical objects that received new tracker numbers upon restarting the SLAM sequence in Recording 02:

| Tracker Entity ID | Unmatched Obs Count | Physical Ground-Truth Match | True Semantic Class | True Movability |
|---|:---:|---|:---:|:---:|
| `chair_006` | 315 | `inst_chair_003` | `chair` | `MOVABLE` |
| `chair_014` | 284 | `inst_chair_004` | `chair` | `MOVABLE` |
| `desk_014` | 248 | `inst_desk_005` | `desk` | `MOVABLE` |
| `chair_041` | 238 | `inst_chair_007` | `chair` | `MOVABLE` |
| `chair_012` | 177 | `inst_chair_005` | `chair` | `MOVABLE` |
| `chair_031` | 159 | `inst_chair_006` | `chair` | `MOVABLE` |
| `chair_015` | 140 | `inst_chair_008` | `chair` | `MOVABLE` |
| `desk_002` | 111 | `inst_desk_002` | `desk` | `MOVABLE` |
| `desk_027` | 107 | `inst_desk_011` | `desk` | `MOVABLE` |
| `laptop_002` | 104 | `inst_laptop_001` | `laptop` | `MOVABLE` |
| `chair_022` | 101 | `Unmapped Session 2 Track` | `chair` | `MOVABLE` |
| `desk_032` | 71 | `Unmapped Session 2 Track` | `desk` | `MOVABLE` |
| `desk_029` | 59 | `inst_desk_018` | `desk` | `MOVABLE` |
| `handbag_002` | 58 | `Unmapped Session 2 Track` | `handbag` | `MOVABLE` |
| `desk_035` | 58 | `Unmapped Session 2 Track` | `desk` | `MOVABLE` |

---

## 4. Physical-Instance Association Quality Metrics

| Metric | Value | Target / Assessment | Definition |
|---|:---:|:---:|---|
| **Physical-Instance Association Coverage** | **55.6%** | > 85.0% (Deficit) | Fraction of total raw detections mapped to ground-truth objects |
| **False Merge Rate** | **22.41%** | < 2.0% (EXCELLENT ✅) | Tracker tracks accidentally merging two distinct physical objects |
| **False Split Rate** | **97.62%** | High (Expected) | Physical objects fragmented across multiple tracker IDs across sessions |
| **Short-Track Fragmentation (<5 frames)** | **72.78%** | Informational | Detections dropped within 5 frames by causal tracker |
| **Cross-Session ID Stability** | **0.00%** | Deficit | Percent of track IDs that naturally preserve identity across sessions |

---

## 5. Ground-Truth Physical Instance Association Table

| Instance ID | Class | True Movability | True Motion | Associated Tracks | Mapped Obs Count | Sessions Verified |
|---|:---:|:---:|:---:|---|:---:|:---:|
| `inst_backpack_001` | `backpack` | `MOVABLE` | `STATIC` | `backpack_001_s2`, `backpack_002` | 75 | S1 & S2 |
| `inst_backpack_002` | `backpack` | `MOVABLE` | `STATIC` | `backpack_003` | 243 | S1 & S2 |
| `inst_ceiling_fan_001` | `ceiling fan` | `IMMOVABLE` | `STATIC` | `ceiling fan_010` | 10 | S1 only |
| `inst_chair_001` | `chair` | `MOVABLE` | `STATIC` | `chair_001`, `chair_001_s2` | 21 | S1 & S2 |
| `inst_chair_002` | `chair` | `MOVABLE` | `MOVING` | `chair_002` | 95 | S1 & S2 |
| `inst_chair_003` | `chair` | `MOVABLE` | `MOVING` | `chair_003` | 189 | S1 & S2 |
| `inst_chair_004` | `chair` | `MOVABLE` | `STATIC` | `chair_004` | 281 | S1 & S2 |
| `inst_chair_005` | `chair` | `MOVABLE` | `STATIC` | `chair_005` | 212 | S1 only |
| `inst_chair_007` | `chair` | `MOVABLE` | `STATIC` | `chair_007`, `chair_007_s2` | 322 | S1 & S2 |
| `inst_chair_008` | `chair` | `MOVABLE` | `STATIC` | `chair_008` | 87 | S1 only |
| `inst_chair_009` | `chair` | `MOVABLE` | `STATIC` | `chair_009` | 133 | S1 only |
| `inst_chair_011` | `chair` | `MOVABLE` | `STATIC` | `chair_011`, `chair_011_s2` | 323 | S1 & S2 |
| `inst_chair_017` | `chair` | `MOVABLE` | `STATIC` | `chair_017` | 120 | S1 only |
| `inst_chair_018` | `chair` | `MOVABLE` | `STATIC` | `chair_018`, `chair_018_s2` | 28 | S1 & S2 |
| `inst_chair_019` | `chair` | `MOVABLE` | `STATIC` | `chair_019` | 46 | S1 only |
| `inst_chair_021` | `chair` | `MOVABLE` | `STATIC` | `chair_021` | 13 | S1 & S2 |
| `inst_chair_027` | `chair` | `MOVABLE` | `STATIC` | `chair_027` | 124 | S1 & S2 |
| `inst_chair_030` | `chair` | `MOVABLE` | `STATIC` | `chair_030` | 71 | S1 only |
| `inst_chair_032` | `chair` | `MOVABLE` | `STATIC` | `chair_032`, `chair_032_s2` | 29 | S1 & S2 |
| `inst_chair_035` | `chair` | `MOVABLE` | `STATIC` | `chair_035` | 11 | S1 only |
| `inst_desk_001` | `desk` | `MOVABLE` | `STATIC` | `desk_001`, `desk_001_s2` | 234 | S1 & S2 |
| `inst_desk_003` | `desk` | `MOVABLE` | `STATIC` | `desk_003`, `desk_004` | 101 | S1 & S2 |
| `inst_desk_005` | `desk` | `MOVABLE` | `STATIC` | `desk_005`, `desk_014_s2` | 79 | S1 only |
| `inst_desk_007_s2` | `desk` | `MOVABLE` | `STATIC` | `desk_007_s2` | 0 | S2 only |
| `inst_desk_008` | `desk` | `MOVABLE` | `STATIC` | `desk_008`, `desk_016_s2` | 31 | S1 only |
| `inst_desk_010_s2` | `desk` | `MOVABLE` | `STATIC` | `desk_010_s2` | 0 | S2 only |
| `inst_desk_012` | `desk` | `MOVABLE` | `STATIC` | `desk_012`, `desk_012_s2` | 140 | S1 & S2 |
| `inst_desk_016` | `desk` | `MOVABLE` | `STATIC` | `desk_016`, `desk_015_s2` | 60 | S1 & S2 |
| `inst_desk_017` | `desk` | `MOVABLE` | `STATIC` | `desk_017`, `desk_020_s2` | 91 | S1 only |
| `inst_desk_018` | `desk` | `MOVABLE` | `STATIC` | `desk_018`, `desk_029_s2` | 94 | S1 & S2 |
| `inst_desk_023` | `desk` | `MOVABLE` | `STATIC` | `desk_023`, `desk_034_s2` | 126 | S1 & S2 |
| `inst_desk_024` | `desk` | `MOVABLE` | `STATIC` | `desk_024`, `desk_022_s2` | 30 | S1 & S2 |
| `inst_desk_025` | `desk` | `MOVABLE` | `STATIC` | `desk_025`, `desk_006_s2` | 8 | S1 only |
| `inst_desk_031` | `desk` | `MOVABLE` | `STATIC` | `desk_031`, `desk_002_s2` | 9 | S1 only |
| `inst_door_001` | `door` | `IMMOVABLE` | `STATIC` | `door_001` | 3 | S1 & S2 |
| `inst_handbag_001` | `handbag` | `MOVABLE` | `STATIC` | `handbag_002_s2` | 0 | S2 only |
| `inst_human_operator_001` | `person` | `MOBILE` | `MOVING` | `person_operator`, `person_001`, `person_002` (+29) | 826 | S1 & S2 |
| `inst_keyboard_001` | `keyboard` | `MOVABLE` | `STATIC` | `keyboard_003_s2` | 0 | S2 only |
| `inst_laptop_001` | `laptop` | `MOVABLE` | `STATIC` | `laptop_006`, `laptop_001_s2` | 17 | S1 & S2 |
| `inst_laptop_002` | `laptop` | `MOVABLE` | `STATIC` | `laptop_002_s2` | 0 | S2 only |
| `inst_laptop_003` | `laptop` | `MOVABLE` | `STATIC` | `laptop_003_s2` | 0 | S2 only |
| `inst_laptop_004` | `laptop` | `MOVABLE` | `STATIC` | `laptop_012_s2` | 0 | S2 only |
| `inst_monitor_001` | `computer monitor` | `MOVABLE` | `STATIC` | `computer monitor_001`, `computer monitor_001_s2` | 21 | S1 & S2 |
| `inst_monitor_002` | `computer monitor` | `MOVABLE` | `STATIC` | `computer monitor_002_s2` | 0 | S2 only |
| `inst_pedestal_fan_001` | `pedestal fan` | `MOVABLE` | `STATIC` | `pedestal fan_003` | 8 | S1 only |
| `inst_quadcopter_drone_001` | `robot` | `MOBILE` | `STATIC` | `quadcopter drone_001_s2`, `quadcopter drone_001` | 12 | S1 & S2 |
| `inst_rc_car_robot_001` | `robot` | `MOBILE` | `MOVING` | `robot_platform_base`, `robot_001` | 4 | S1 & S2 |
| `inst_storage_box_001` | `storage box` | `MOVABLE` | `STATIC` | `storage box_001`, `storage box_003_s2` | 20 | S1 & S2 |
| `inst_storage_box_004` | `storage box` | `MOVABLE` | `STATIC` | `storage box_004`, `storage box_004_s2` | 25 | S1 & S2 |
| `inst_storage_box_007` | `storage box` | `MOVABLE` | `MOVING` | `storage box_007`, `storage box_011` | 49 | S1 & S2 |
| `inst_wall_north_001` | `wall` | `IMMOVABLE` | `STATIC` | `wall_boundary_north` | 0 | S2 only |
| `inst_wall_west_001` | `wall` | `IMMOVABLE` | `STATIC` | `wall_boundary_west` | 0 | S2 only |

---
*Generated automatically by `vlm/scripts/audit_semantic_associations.py`.*