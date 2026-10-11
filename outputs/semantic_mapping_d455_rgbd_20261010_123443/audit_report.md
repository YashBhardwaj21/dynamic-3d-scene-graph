# Offline Semantic Mapping Audit Report

**Sequence**: `d455_rgbd_20261010_123443`
**Timestamp**: 2026-10-11 19:12:56
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
- **Origin**: `(-4.700, -5.650) m`
- **Dimensions**: `10.60 x 11.15 m` @ `0.05 m/cell` (212 x 223 cells)
- **Free Cells**: `6560 (13.88%)`
- **Occupied Cells**: `4437 (9.39%)`
- **Unknown Cells**: `36279 (76.74%)`
- **Inflated Navigation Occupied Cells**: `12896 (27.28%)`
- **Robot Radius / Safety Margin**: `0.18 m` / `0.05 m`

---

## 4. Semantic Objects Summary
- **Confirmed Static Entities**: `26`
- **Quarantined Candidates**: `33`

| Entity ID | Class | Observations | Frame Span | Dimensions (WxLxH m) | Spread $\sigma$ (m) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `backpack_001` | backpack | 89 | 780 | `0.67x0.99x0.42` | `0.085` | **CONFIRMED** |
| `backpack_003` | backpack | 78 | 770 | `0.53x0.62x0.40` | `0.058` | **CONFIRMED** |
| `backpack_007` | backpack | 24 | 240 | `0.30x0.41x0.42` | `0.055` | **CONFIRMED** |
| `backpack_008` | backpack | 60 | 650 | `0.43x1.38x0.42` | `0.051` | **CONFIRMED** |
| `backpack_009` | backpack | 30 | 590 | `0.42x0.32x0.43` | `0.025` | **CONFIRMED** |
| `chair_001` | chair | 81 | 690 | `1.23x0.67x0.85` | `0.199` | **CONFIRMED** |
| `chair_002` | chair | 77 | 780 | `0.96x0.66x0.84` | `0.209` | **CONFIRMED** |
| `chair_005` | chair | 39 | 460 | `1.28x0.38x0.79` | `0.119` | **CONFIRMED** |
| `chair_008` | chair | 16 | 240 | `1.06x0.26x0.71` | `0.167` | **CONFIRMED** |
| `chair_010` | chair | 47 | 570 | `0.63x0.93x0.81` | `0.233` | **CONFIRMED** |
| `chair_013` | chair | 51 | 500 | `1.10x1.07x0.92` | `0.129` | **CONFIRMED** |
| `chair_014` | chair | 8 | 480 | `0.48x0.53x0.65` | `0.065` | **CONFIRMED** |
| `chair_016` | chair | 13 | 580 | `0.47x0.95x0.84` | `0.155` | **CONFIRMED** |
| `desk_001` | desk | 27 | 750 | `1.11x1.10x0.66` | `0.226` | **CONFIRMED** |
| `desk_002` | desk | 22 | 340 | `1.34x0.77x0.75` | `0.163` | **CONFIRMED** |
| `desk_006` | desk | 8 | 250 | `2.62x1.50x0.79` | `0.209` | **CONFIRMED** |
| `desk_007` | desk | 21 | 320 | `2.58x1.59x1.18` | `0.126` | **CONFIRMED** |
| `desk_009` | desk | 19 | 480 | `0.31x0.61x0.27` | `0.089` | **CONFIRMED** |
| `desk_010` | desk | 14 | 360 | `0.23x1.15x0.69` | `0.118` | **CONFIRMED** |
| `door_003` | door | 58 | 720 | `1.14x2.33x2.33` | `0.259` | **CONFIRMED** |
| `handbag_001` | handbag | 79 | 720 | `0.35x0.66x0.44` | `0.125` | **CONFIRMED** |
| `laptop_001` | laptop | 32 | 740 | `0.58x0.23x0.33` | `0.170` | **CONFIRMED** |
| `laptop_003` | laptop | 9 | 130 | `0.51x0.16x0.27` | `0.010` | **CONFIRMED** |
| `laptop_004` | laptop | 13 | 210 | `0.52x0.15x0.28` | `0.020` | **CONFIRMED** |
| `storage box_003` | storage box | 11 | 320 | `0.51x0.14x0.24` | `0.279` | **CONFIRMED** |
| `storage box_005` | storage box | 12 | 270 | `0.39x0.22x0.32` | `0.031` | **CONFIRMED** |
| `backpack_002` | backpack | 3 | 190 | `0.35x0.40x0.37` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `backpack_005` | backpack | 1 | 0 | `0.10x0.08x0.15` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `backpack_006` | backpack | 1 | 0 | `0.10x0.08x0.15` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `ceiling fan_001` | ceiling fan | 3 | 20 | `0.21x1.57x0.16` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `ceiling fan_002` | ceiling fan | 4 | 180 | `0.62x1.05x0.77` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `ceiling fan_003` | ceiling fan | 1 | 0 | `0.04x0.03x0.04` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_003` | chair | 7 | 350 | `1.32x0.57x0.60` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `chair_007` | chair | 1 | 0 | `0.26x0.24x0.11` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_009` | chair | 2 | 80 | `0.39x0.16x0.18` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `chair_011` | chair | 7 | 370 | `0.34x0.42x0.45` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `chair_015` | chair | 1 | 0 | `0.19x0.53x0.48` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_017` | chair | 1 | 0 | `0.18x0.69x0.46` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `computer monitor_001` | computer monitor | 4 | 190 | `0.52x0.14x0.26` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `computer monitor_002` | computer monitor | 5 | 70 | `0.52x0.16x0.29` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `desk_003` | desk | 1 | 0 | `0.07x0.12x0.25` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_004` | desk | 1 | 0 | `1.32x1.21x0.37` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_005` | desk | 1 | 0 | `1.90x0.94x0.73` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_008` | desk | 7 | 200 | `0.13x0.33x0.72` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `desk_011` | desk | 6 | 70 | `0.10x0.18x0.69` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `door_001` | door | 7 | 130 | `0.14x0.26x0.51` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `door_004` | door | 1 | 0 | `0.05x0.06x0.72` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `laptop_002` | laptop | 3 | 30 | `0.31x0.17x0.13` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `person_001` | person | 3 | 20 | `1.11x0.27x1.45` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_002` | person | 9 | 100 | `1.55x0.73x1.57` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_003` | person | 5 | 70 | `0.11x0.17x0.12` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_004` | person | 2 | 20 | `0.39x0.25x0.44` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `storage box_001` | storage box | 2 | 80 | `0.06x0.12x0.08` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `storage box_002` | storage box | 6 | 190 | `0.06x0.23x0.09` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `storage box_004` | storage box | 5 | 290 | `0.13x0.30x0.70` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `whiteboard_001` | whiteboard | 1 | 0 | `0.27x0.21x0.27` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `whiteboard_002` | whiteboard | 1 | 0 | `0.37x0.21x0.31` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `whiteboard_003` | whiteboard | 1 | 0 | `1.02x0.86x1.85` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `whiteboard_004` | whiteboard | 3 | 20 | `0.21x0.17x0.29` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |

---

## 5. Pairwise Clearances
Pairwise clearances computed with derived measurement uncertainty:
$\sigma_{AB} = \sqrt{\sigma_A^2 + \sigma_B^2 + 0.015^2}$
