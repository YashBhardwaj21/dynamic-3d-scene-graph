# Offline Semantic Mapping Audit Report

**Sequence**: `d455_rgbd_20261010_123443`
**Timestamp**: 2026-10-11 14:46:56
**Pipeline Mode**: `fresh`
**Output Path**: `/mnt/c/Users/Yash Bhardwaj/Desktop/perception/outputs/semantic_mapping_d455_rgbd_20261010_123443`

---

## 1. Sequence Provenance & Coverage
- **Total RGB Frames**: `2492`
- **Total Depth Frames**: `2492`
- **Synchronized RGB-D Pairs**: `2492`
- **Sampling Policy**: `step` (step=10, fps=5.0)
- **Sampled Frames Processed**: `250`
- **Input Coverage**: `10.03%` of all available sequence frames
- **Pose Alignment Coverage**: `100.0%` of sampled frames matched to SLAM poses

---

## 2. Floor Coordinate System & Transform
- **Frame Convention**: ROS REP-103 (`floor_map` frame: X=Forward, Y=Left, Z=Height above floor)
- **Fitted Floor Plane**:
  - Upward Normal: `[0.1627, -0.9839, -0.0743]`
  - Plane Distance $d$: `0.5357 m`
  - Inliers: `25071/36092 (69.46%)`
  - Residual Mean/Std: `6.2 mm` / `5.7 mm`
- **Transform**: Full $4 \times 4$ matrix $T_{\text{floor\_world}}$ stored in `floor_plane.json`.

---

## 3. Dynamic Occupancy Grid & Obstacle Clearances
- **Origin**: `(-2.050, -2.100) m`
- **Dimensions**: `4.25 x 4.05 m` @ `0.05 m/cell` (85 x 81 cells)
- **Free Cells**: `3553 (51.6%)`
- **Occupied Cells**: `1909 (27.73%)`
- **Unknown Cells**: `1423 (20.67%)`
- **Inflated Navigation Occupied Cells**: `4543 (65.98%)`
- **Robot Radius / Safety Margin**: `0.18 m` / `0.05 m`

---

## 4. Semantic Objects Summary
- **Confirmed Static Entities**: `19`
- **Quarantined Candidates**: `15`

| Entity ID | Class | Observations | Frame Span | Dimensions (WxLxH m) | Spread $\sigma$ (m) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `backpack_001` | backpack | 180 | 810 | `0.95x1.00x0.74` | `0.247` | **CONFIRMED** |
| `backpack_006` | backpack | 24 | 240 | `0.30x0.41x0.42` | `0.055` | **CONFIRMED** |
| `backpack_007` | backpack | 60 | 650 | `0.43x1.38x0.42` | `0.051` | **CONFIRMED** |
| `chair_001` | chair | 76 | 690 | `1.21x0.67x0.84` | `0.139` | **CONFIRMED** |
| `chair_002` | chair | 82 | 780 | `0.97x0.68x0.84` | `0.257` | **CONFIRMED** |
| `chair_003` | desk | 30 | 550 | `1.71x0.82x0.75` | `0.338` | **CONFIRMED** |
| `chair_005` | chair | 39 | 460 | `1.30x0.38x0.79` | `0.119` | **CONFIRMED** |
| `chair_008` | chair | 16 | 240 | `1.06x0.26x0.71` | `0.167` | **CONFIRMED** |
| `chair_010` | chair | 47 | 570 | `0.63x0.94x0.81` | `0.233` | **CONFIRMED** |
| `chair_011` | chair | 54 | 530 | `1.15x1.16x0.92` | `0.180` | **CONFIRMED** |
| `chair_015` | backpack | 58 | 610 | `0.49x0.67x0.66` | `0.135` | **CONFIRMED** |
| `chair_016` | chair | 40 | 580 | `0.48x1.19x0.84` | `0.269` | **CONFIRMED** |
| `desk_001` | desk | 42 | 790 | `1.12x0.93x1.08` | `0.289` | **CONFIRMED** |
| `desk_004` | desk | 11 | 350 | `2.59x1.50x0.79` | `0.293` | **CONFIRMED** |
| `door_002` | door | 58 | 710 | `1.24x1.84x2.30` | `0.258` | **CONFIRMED** |
| `handbag_001` | handbag | 79 | 720 | `0.35x0.66x0.44` | `0.125` | **CONFIRMED** |
| `laptop_001` | laptop | 33 | 740 | `0.61x0.28x0.33` | `0.175` | **CONFIRMED** |
| `storage box_001` | door | 9 | 270 | `0.40x0.26x0.51` | `0.148` | **CONFIRMED** |
| `storage box_003` | desk | 51 | 330 | `2.43x1.59x1.23` | `0.325` | **CONFIRMED** |
| `backpack_005` | backpack | 1 | 0 | `0.10x0.08x0.15` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `ceiling fan_001` | ceiling fan | 3 | 20 | `0.21x1.57x0.16` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `ceiling fan_002` | ceiling fan | 4 | 180 | `0.62x1.05x0.77` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `chair_006` | chair | 1 | 0 | `0.26x0.24x0.11` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_009` | laptop | 5 | 90 | `0.39x0.18x0.18` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `chair_013` | chair | 6 | 180 | `0.59x0.40x0.84` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `desk_002` | desk | 1 | 0 | `0.07x0.12x0.25` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_003` | desk | 1 | 0 | `1.32x1.21x0.37` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_007` | desk | 6 | 70 | `0.10x0.18x0.69` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `door_001` | door | 2 | 10 | `0.40x0.92x1.63` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `person_001` | person | 3 | 20 | `1.11x0.27x1.45` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_002` | person | 9 | 100 | `1.55x0.73x1.57` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_003` | person | 5 | 70 | `0.11x0.17x0.12` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_004` | person | 2 | 20 | `0.39x0.25x0.44` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `storage box_002` | storage box | 6 | 190 | `0.06x0.23x0.09` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |

---

## 5. Pairwise Clearances
Pairwise clearances computed with derived measurement uncertainty:
$\sigma_{AB} = \sqrt{\sigma_A^2 + \sigma_B^2 + 0.015^2}$
