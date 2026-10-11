# Offline Semantic Mapping Audit Report

**Sequence**: `desk_run_20261009_161552`
**Timestamp**: 2026-10-11 19:03:02
**Pipeline Mode**: `fresh`
**Output Path**: `/mnt/c/Users/Yash Bhardwaj/Desktop/perception/outputs/semantic_mapping_desk_run_20261009_161552`

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
- **Origin**: `(-5.250, -4.000) m`
- **Dimensions**: `13.15 x 13.10 m` @ `0.05 m/cell` (263 x 262 cells)
- **Free Cells**: `10579 (15.35%)`
- **Occupied Cells**: `7370 (10.7%)`
- **Unknown Cells**: `50957 (73.95%)`
- **Inflated Navigation Occupied Cells**: `20106 (29.18%)`
- **Robot Radius / Safety Margin**: `0.18 m` / `0.05 m`

---

## 4. Semantic Objects Summary
- **Confirmed Static Entities**: `26`
- **Quarantined Candidates**: `73`

| Entity ID | Class | Observations | Frame Span | Dimensions (WxLxH m) | Spread $\sigma$ (m) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ceiling fan_003` | ceiling fan | 21 | 155 | `1.48x1.41x0.78` | `0.361` | **CONFIRMED** |
| `ceiling fan_006` | ceiling fan | 8 | 35 | `0.83x0.88x0.72` | `0.195` | **CONFIRMED** |
| `chair_001` | chair | 10 | 100 | `0.46x0.18x0.40` | `0.224` | **CONFIRMED** |
| `chair_002` | chair | 35 | 180 | `1.29x0.82x0.96` | `0.132` | **CONFIRMED** |
| `chair_003` | chair | 9 | 310 | `1.10x0.64x0.80` | `0.207` | **CONFIRMED** |
| `chair_004` | chair | 42 | 245 | `0.88x0.55x0.68` | `0.103` | **CONFIRMED** |
| `chair_005` | chair | 44 | 260 | `0.89x0.64x0.72` | `0.076` | **CONFIRMED** |
| `chair_010` | chair | 34 | 180 | `1.31x0.70x0.79` | `0.180` | **CONFIRMED** |
| `chair_016` | chair | 63 | 270 | `1.03x1.59x0.89` | `0.159` | **CONFIRMED** |
| `chair_018` | chair | 12 | 140 | `0.96x0.60x0.73` | `0.302` | **CONFIRMED** |
| `chair_024` | chair | 47 | 185 | `1.53x1.07x0.87` | `0.207` | **CONFIRMED** |
| `chair_026` | chair | 63 | 215 | `1.85x0.95x0.86` | `0.231` | **CONFIRMED** |
| `chair_029` | chair | 13 | 145 | `0.37x0.20x0.34` | `0.082` | **CONFIRMED** |
| `chair_030` | chair | 27 | 130 | `2.44x1.80x1.41` | `0.190` | **CONFIRMED** |
| `desk_001` | desk | 50 | 395 | `2.14x2.13x1.01` | `0.273` | **CONFIRMED** |
| `desk_002` | desk | 11 | 65 | `1.06x0.76x0.63` | `0.255` | **CONFIRMED** |
| `desk_003` | desk | 14 | 120 | `1.16x0.89x0.66` | `0.296` | **CONFIRMED** |
| `desk_004` | desk | 51 | 290 | `0.77x0.95x0.79` | `0.247` | **CONFIRMED** |
| `desk_005` | desk | 22 | 105 | `0.24x0.30x0.36` | `0.054` | **CONFIRMED** |
| `desk_006` | desk | 28 | 265 | `1.20x1.39x0.92` | `0.226` | **CONFIRMED** |
| `desk_009` | desk | 51 | 245 | `0.69x1.22x0.79` | `0.282` | **CONFIRMED** |
| `desk_010` | desk | 9 | 50 | `0.35x0.41x0.34` | `0.060` | **CONFIRMED** |
| `desk_013` | desk | 61 | 300 | `1.84x0.88x0.84` | `0.224` | **CONFIRMED** |
| `desk_015` | desk | 27 | 130 | `1.46x1.13x0.53` | `0.240` | **CONFIRMED** |
| `laptop_003` | laptop | 11 | 120 | `0.44x0.53x0.28` | `0.098` | **CONFIRMED** |
| `storage box_001` | storage box | 11 | 55 | `0.23x0.48x0.37` | `0.186` | **CONFIRMED** |
| `backpack_001` | handbag | 4 | 15 | `0.53x0.51x0.59` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `ceiling fan_001` | ceiling fan | 1 | 0 | `0.20x0.12x0.04` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `ceiling fan_002` | ceiling fan | 1 | 0 | `0.01x0.02x0.01` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `ceiling fan_004` | ceiling fan | 1 | 0 | `0.18x0.09x0.06` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `ceiling fan_005` | ceiling fan | 1 | 0 | `0.04x0.03x0.02` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `ceiling fan_007` | ceiling fan | 2 | 140 | `1.19x1.06x0.47` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `ceiling fan_008` | ceiling fan | 2 | 5 | `0.31x0.29x0.25` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `ceiling fan_009` | ceiling fan | 2 | 5 | `0.66x0.45x0.27` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `chair_006` | chair | 1 | 0 | `1.48x0.61x0.78` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_007` | chair | 7 | 45 | `1.08x0.88x0.68` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `chair_008` | chair | 2 | 5 | `0.40x0.17x0.37` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `chair_009` | chair | 4 | 160 | `2.13x0.82x1.07` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `chair_012` | chair | 1 | 0 | `0.30x0.30x0.27` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_013` | chair | 1 | 0 | `1.59x0.32x1.11` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_014` | chair | 3 | 10 | `1.21x0.48x0.67` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `chair_015` | chair | 1 | 0 | `0.89x0.33x0.25` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_019` | chair | 1 | 0 | `0.17x0.13x0.13` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_020` | chair | 5 | 20 | `0.10x0.04x0.16` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `chair_021` | chair | 5 | 40 | `0.04x0.09x0.17` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `chair_022` | chair | 3 | 40 | `1.17x0.54x0.52` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `chair_023` | chair | 1 | 0 | `0.66x0.66x0.48` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_028` | chair | 5 | 25 | `0.58x0.19x0.53` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `chair_031` | chair | 1 | 0 | `0.13x0.06x0.27` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `computer monitor_001` | computer monitor | 3 | 10 | `0.30x0.34x0.26` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `desk_007` | desk | 7 | 75 | `0.37x0.41x0.14` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `desk_008` | desk | 4 | 60 | `0.42x0.83x0.72` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `desk_012` | desk | 1 | 0 | `0.28x0.72x0.67` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_014` | desk | 1 | 0 | `0.39x0.17x0.18` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_016` | desk | 5 | 40 | `0.83x0.85x0.47` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `desk_017` | desk | 6 | 25 | `0.53x0.51x0.36` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `desk_018` | desk | 1 | 0 | `0.35x0.21x0.24` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `handbag_001` | handbag | 1 | 0 | `0.25x0.23x0.26` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `handbag_002` | handbag | 1 | 0 | `0.05x0.06x0.11` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `handbag_003` | handbag | 2 | 5 | `0.32x0.29x0.27` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `handbag_004` | handbag | 4 | 20 | `0.07x0.09x0.11` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `handbag_005` | handbag | 1 | 0 | `0.47x0.34x0.11` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `handbag_006` | handbag | 2 | 60 | `1.97x1.14x0.75` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `handbag_007` | handbag | 7 | 50 | `0.31x0.40x0.39` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `keyboard_001` | keyboard | 1 | 0 | `0.11x0.29x0.04` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `laptop_001` | laptop | 3 | 15 | `0.13x0.16x0.19` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `laptop_002` | laptop | 5 | 55 | `0.63x0.24x0.01` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `laptop_004` | laptop | 3 | 15 | `0.40x0.40x0.26` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `laptop_005` | laptop | 4 | 20 | `0.39x1.17x0.14` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `person_001` | person | 14 | 280 | `0.60x0.61x0.35` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_002` | person | 9 | 280 | `0.44x0.37x0.35` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_003` | person | 3 | 10 | `1.13x0.71x1.65` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_004` | person | 3 | 10 | `1.07x0.80x1.71` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_005` | person | 5 | 20 | `1.25x0.84x1.61` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_006` | person | 1 | 0 | `0.07x0.11x0.03` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_007` | person | 2 | 10 | `0.10x0.08x0.06` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_008` | person | 14 | 250 | `0.50x1.21x0.36` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_009` | person | 2 | 10 | `0.04x0.04x0.01` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_010` | person | 1 | 0 | `0.13x0.05x0.29` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_011` | person | 10 | 140 | `0.62x0.69x0.49` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_012` | person | 11 | 90 | `0.24x0.34x0.15` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_013` | person | 1 | 0 | `0.11x0.08x0.39` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_014` | person | 8 | 55 | `0.25x1.02x0.28` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_015` | person | 5 | 20 | `0.61x0.88x0.39` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_016` | person | 1 | 0 | `0.06x0.15x0.09` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `projector_001` | projector | 1 | 0 | `0.28x0.26x0.13` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `projector_002` | projector | 2 | 160 | `0.20x0.17x0.23` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `quadcopter drone_001` | quadcopter drone | 1 | 0 | `0.39x0.51x0.41` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `quadcopter drone_002` | quadcopter drone | 1 | 0 | `0.26x0.22x0.28` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `robot_001` | robot | 3 | 35 | `0.31x0.53x0.38` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `robot_002` | robot | 7 | 515 | `0.42x0.47x0.28` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `robot_003` | robot | 4 | 55 | `0.22x0.26x0.18` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `robot_004` | robot | 3 | 15 | `0.23x0.27x0.13` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `robot_005` | robot | 3 | 15 | `0.36x0.30x0.29` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `robot_006` | robot | 4 | 30 | `0.45x0.21x0.25` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `storage box_002` | storage box | 4 | 45 | `0.12x0.17x0.19` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `storage box_003` | storage box | 2 | 75 | `0.20x0.23x0.08` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `storage box_004` | storage box | 2 | 5 | `0.22x0.28x0.11` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `storage box_005` | storage box | 2 | 5 | `0.04x0.08x0.03` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |

---

## 5. Pairwise Clearances
Pairwise clearances computed with derived measurement uncertainty:
$\sigma_{AB} = \sqrt{\sigma_A^2 + \sigma_B^2 + 0.015^2}$
