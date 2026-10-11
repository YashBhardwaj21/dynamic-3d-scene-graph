# Offline Semantic Mapping Audit Report

**Sequence**: `desk_run_20261009_161552`
**Timestamp**: 2026-10-11 14:37:59
**Pipeline Mode**: `fresh`
**Output Path**: `/mnt/c/Users/Yash Bhardwaj/Desktop/perception/outputs/semantic_mapping_desk_run_161552`

---

## 1. Sequence Provenance & Coverage
- **Total RGB Frames**: `1043`
- **Total Depth Frames**: `1046`
- **Synchronized RGB-D Pairs**: `1043`
- **Sampling Policy**: `step` (step=5, fps=5.0)
- **Sampled Frames Processed**: `209`
- **Input Coverage**: `20.04%` of all available sequence frames
- **Pose Alignment Coverage**: `100.0%` of sampled frames matched to SLAM poses

---

## 2. Floor Coordinate System & Transform
- **Frame Convention**: ROS REP-103 (`floor_map` frame: X=Forward, Y=Left, Z=Height above floor)
- **Fitted Floor Plane**:
  - Upward Normal: `[-0.0582, -0.9006, 0.4307]`
  - Plane Distance $d$: `0.8886 m`
  - Inliers: `9120/30061 (30.34%)`
  - Residual Mean/Std: `11.9 mm` / `8.7 mm`
- **Transform**: Full $4 \times 4$ matrix $T_{\text{floor\_world}}$ stored in `floor_plane.json`.

---

## 3. Dynamic Occupancy Grid & Obstacle Clearances
- **Origin**: `(-2.450, -1.400) m`
- **Dimensions**: `6.15 x 6.80 m` @ `0.05 m/cell` (123 x 136 cells)
- **Free Cells**: `6759 (40.41%)`
- **Occupied Cells**: `5097 (30.47%)`
- **Unknown Cells**: `4872 (29.12%)`
- **Inflated Navigation Occupied Cells**: `12187 (72.85%)`
- **Robot Radius / Safety Margin**: `0.18 m` / `0.05 m`

---

## 4. Semantic Objects Summary
- **Confirmed Static Entities**: `24`
- **Quarantined Candidates**: `42`

| Entity ID | Class | Observations | Frame Span | Dimensions (WxLxH m) | Spread $\sigma$ (m) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ceiling fan_002` | ceiling fan | 21 | 155 | `1.48x1.41x0.78` | `0.361` | **CONFIRMED** |
| `ceiling fan_003` | ceiling fan | 8 | 35 | `0.88x0.93x0.77` | `0.195` | **CONFIRMED** |
| `chair_001` | chair | 11 | 100 | `0.46x0.42x0.40` | `0.256` | **CONFIRMED** |
| `chair_002` | chair | 35 | 180 | `1.29x0.82x0.96` | `0.132` | **CONFIRMED** |
| `chair_003` | desk | 79 | 395 | `1.56x1.68x1.21` | `0.385` | **CONFIRMED** |
| `chair_004` | chair | 42 | 245 | `0.91x0.56x0.68` | `0.103` | **CONFIRMED** |
| `chair_005` | chair | 45 | 260 | `0.92x0.64x0.79` | `0.080` | **CONFIRMED** |
| `chair_007` | chair | 34 | 180 | `1.31x0.70x0.79` | `0.180` | **CONFIRMED** |
| `chair_012` | chair | 75 | 560 | `1.03x1.72x1.13` | `0.285` | **CONFIRMED** |
| `chair_013` | chair | 14 | 150 | `1.07x0.78x0.92` | `0.381` | **CONFIRMED** |
| `chair_018` | desk | 8 | 45 | `0.22x0.37x0.37` | `0.060` | **CONFIRMED** |
| `chair_021` | chair | 87 | 215 | `1.78x1.09x0.86` | `0.249` | **CONFIRMED** |
| `chair_023` | chair | 10 | 125 | `1.28x0.29x0.87` | `0.375` | **CONFIRMED** |
| `chair_025` | chair | 23 | 110 | `2.44x1.79x1.41` | `0.147` | **CONFIRMED** |
| `desk_002` | desk | 39 | 205 | `1.31x1.19x1.01` | `0.417` | **CONFIRMED** |
| `desk_003` | desk | 53 | 300 | `0.77x0.97x0.79` | `0.274` | **CONFIRMED** |
| `desk_004` | desk | 22 | 105 | `0.24x0.30x0.40` | `0.054` | **CONFIRMED** |
| `desk_005` | desk | 28 | 265 | `1.20x1.39x0.92` | `0.226` | **CONFIRMED** |
| `desk_008` | desk | 99 | 245 | `1.44x1.39x0.87` | `0.315` | **CONFIRMED** |
| `desk_012` | desk | 59 | 290 | `1.87x0.88x0.84` | `0.205` | **CONFIRMED** |
| `desk_013` | handbag | 8 | 265 | `0.34x0.41x0.39` | `0.220` | **CONFIRMED** |
| `desk_014` | desk | 26 | 125 | `1.36x1.07x0.44` | `0.215` | **CONFIRMED** |
| `laptop_002` | laptop | 12 | 120 | `0.47x0.53x0.28` | `0.098` | **CONFIRMED** |
| `robot_001` | desk | 13 | 105 | `1.48x0.90x1.12` | `0.311` | **CONFIRMED** |
| `backpack_001` | chair | 3 | 10 | `0.43x0.51x0.59` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `ceiling fan_001` | ceiling fan | 1 | 0 | `0.20x0.12x0.04` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `ceiling fan_004` | ceiling fan | 2 | 140 | `1.19x1.06x0.47` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `ceiling fan_005` | ceiling fan | 2 | 5 | `0.66x0.45x0.27` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `chair_006` | chair | 4 | 160 | `2.13x0.82x1.07` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `chair_009` | chair | 1 | 0 | `1.59x0.32x1.11` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_010` | chair | 3 | 10 | `1.21x0.48x0.67` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `chair_011` | chair | 1 | 0 | `0.89x0.33x0.25` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_014` | chair | 5 | 635 | `0.68x0.66x0.79` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `chair_015` | chair | 5 | 20 | `0.10x0.04x0.16` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `chair_016` | chair | 5 | 40 | `0.04x0.09x0.18` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `chair_017` | chair | 4 | 20 | `0.98x0.55x0.76` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `desk_006` | desk | 7 | 75 | `0.37x0.41x0.14` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `desk_009` | desk | 3 | 25 | `0.22x0.47x0.53` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `desk_011` | desk | 1 | 0 | `0.28x0.72x0.67` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_015` | desk | 1 | 0 | `0.35x0.21x0.25` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_016` | desk | 1 | 0 | `0.41x0.51x0.35` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `handbag_001` | handbag | 3 | 10 | `0.69x0.41x0.16` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `handbag_002` | handbag | 1 | 0 | `0.14x0.10x0.16` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `handbag_003` | chair | 6 | 85 | `1.83x1.14x0.74` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `laptop_001` | laptop | 5 | 30 | `1.51x0.72x1.09` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `laptop_004` | laptop | 4 | 20 | `0.39x1.20x0.14` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `person_001` | person | 14 | 280 | `0.69x0.64x0.37` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_002` | person | 9 | 280 | `0.45x0.37x0.36` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_003` | person | 3 | 10 | `1.13x0.71x1.65` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_004` | person | 3 | 10 | `1.07x0.80x1.71` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_005` | person | 5 | 20 | `1.25x0.84x1.61` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_006` | person | 1 | 0 | `0.07x0.11x0.03` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_007` | person | 2 | 10 | `0.10x0.08x0.06` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_008` | person | 14 | 250 | `0.50x1.26x0.36` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_009` | person | 2 | 10 | `0.04x0.04x0.01` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_010` | person | 1 | 0 | `0.13x0.05x0.29` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_011` | person | 10 | 140 | `0.62x0.69x0.49` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_012` | person | 11 | 90 | `0.24x0.34x0.15` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_013` | person | 1 | 0 | `0.11x0.08x0.39` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_014` | person | 8 | 55 | `0.27x1.04x0.28` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_015` | person | 5 | 20 | `0.61x0.88x0.39` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_016` | person | 1 | 0 | `0.06x0.15x0.09` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `projector_001` | projector | 1 | 0 | `0.28x0.26x0.13` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `projector_002` | projector | 2 | 160 | `0.20x0.17x0.23` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `quadcopter drone_001` | robot | 6 | 35 | `0.37x0.30x0.33` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `storage box_003` | storage box | 2 | 5 | `0.04x0.08x0.03` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |

---

## 5. Pairwise Clearances
Pairwise clearances computed with derived measurement uncertainty:
$\sigma_{AB} = \sqrt{\sigma_A^2 + \sigma_B^2 + 0.015^2}$
