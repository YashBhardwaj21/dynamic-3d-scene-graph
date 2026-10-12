# Offline Semantic Mapping Audit Report

**Sequence**: `my_desk_recording_02`
**Timestamp**: 2026-10-12 16:42:53
**Pipeline Mode**: `fresh`
**Output Path**: `/mnt/c/Users/Yash Bhardwaj/Desktop/perception/outputs/semantic_mapping_my_desk_recording_02`

---

## 1. Sequence Provenance & Coverage
- **Total RGB Frames**: `5287`
- **Total Depth Frames**: `5288`
- **Synchronized RGB-D Pairs**: `5283`
- **Sampling Policy**: `step` (step=5, fps=5.0)
- **Sampled Frames Processed**: `1057`
- **Input Coverage**: `20.01%` of all available sequence frames
- **Pose Alignment Coverage**: `100.0%` of sampled frames matched to SLAM poses

---

## 2. Floor Coordinate System & Transform
- **Frame Convention**: ROS REP-103 (`floor_map` frame: X=Forward, Y=Left, Z=Height above floor)
- **Fitted Floor Plane**:
  - Upward Normal: `[-0.1004, -0.9913, 0.0853]`
  - Plane Distance $d$: `0.5241 m`
  - Inliers: `26980/35389 (76.24%)`
  - Residual Mean/Std: `6.8 mm` / `6.6 mm`
- **Transform**: Full $4 \times 4$ matrix $T_{\text{floor\_world}}$ stored in `floor_plane.json`.

---

## 3. Dynamic Occupancy Grid & Obstacle Clearances
- **Origin**: `(-4.350, -9.150) m`
- **Dimensions**: `11.05 x 11.85 m` @ `0.05 m/cell` (221 x 237 cells)
- **Free Cells**: `10322 (19.71%)`
- **Occupied Cells**: `12749 (24.34%)`
- **Unknown Cells**: `29306 (55.95%)`
- **Inflated Navigation Occupied Cells**: `27260 (52.05%)`
- **Robot Radius / Safety Margin**: `0.18 m` / `0.05 m`

---

## 4. Semantic Objects Summary
- **Confirmed Static Entities**: `69`
- **Quarantined Candidates**: `145`

| Entity ID | Class | Observations | Frame Span | Dimensions (WxLxH m) | Spread $\sigma$ (m) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `backpack_001` | backpack | 8 | 35 | `0.35x0.20x0.40` | `0.187` | **CONFIRMED** |
| `chair_001` | chair | 9 | 5175 | `0.94x0.60x0.80` | `0.298` | **CONFIRMED** |
| `chair_004` | chair | 228 | 3460 | `1.05x1.54x0.86` | `0.168` | **CONFIRMED** |
| `chair_006` | chair | 315 | 2860 | `0.92x1.57x0.90` | `0.199` | **CONFIRMED** |
| `chair_007` | chair | 163 | 2830 | `0.77x1.46x0.89` | `0.225` | **CONFIRMED** |
| `chair_011` | chair | 214 | 2675 | `0.86x0.85x0.93` | `0.169` | **CONFIRMED** |
| `chair_012` | chair | 177 | 3780 | `0.94x1.20x0.90` | `0.228` | **CONFIRMED** |
| `chair_014` | chair | 284 | 1100 | `0.74x2.06x0.86` | `0.249` | **CONFIRMED** |
| `chair_015` | chair | 134 | 1090 | `0.91x2.39x0.84` | `0.334` | **CONFIRMED** |
| `chair_018` | chair | 8 | 1270 | `2.74x3.45x2.34` | `0.484` | **CONFIRMED** |
| `chair_022` | chair | 100 | 420 | `0.86x1.06x0.78` | `0.154` | **CONFIRMED** |
| `chair_025` | chair | 13 | 85 | `0.77x0.61x0.52` | `0.086` | **CONFIRMED** |
| `chair_031` | chair | 153 | 875 | `1.10x1.17x0.92` | `0.171` | **CONFIRMED** |
| `chair_032` | chair | 8 | 445 | `1.46x1.04x0.72` | `0.329` | **CONFIRMED** |
| `chair_036` | chair | 11 | 850 | `0.51x0.87x0.34` | `0.186` | **CONFIRMED** |
| `chair_041` | chair | 231 | 1300 | `0.81x1.38x0.88` | `0.166` | **CONFIRMED** |
| `chair_046` | chair | 26 | 175 | `0.39x0.90x0.09` | `0.194` | **CONFIRMED** |
| `chair_049` | chair | 33 | 1600 | `0.41x0.56x0.61` | `0.176` | **CONFIRMED** |
| `computer monitor_001` | computer monitor | 10 | 3665 | `0.32x0.68x0.41` | `0.104` | **CONFIRMED** |
| `computer monitor_002` | computer monitor | 21 | 1080 | `0.37x0.53x0.39` | `0.080` | **CONFIRMED** |
| `computer monitor_003` | computer monitor | 25 | 3245 | `0.34x0.60x0.39` | `0.084` | **CONFIRMED** |
| `computer monitor_005` | computer monitor | 46 | 960 | `0.32x0.62x0.40` | `0.079` | **CONFIRMED** |
| `computer monitor_012` | computer monitor | 12 | 215 | `0.51x1.37x0.38` | `0.138` | **CONFIRMED** |
| `desk_001` | desk | 197 | 5275 | `1.37x1.09x0.94` | `0.178` | **CONFIRMED** |
| `desk_002` | desk | 110 | 5280 | `1.01x1.18x0.85` | `0.228` | **CONFIRMED** |
| `desk_003` | desk | 43 | 800 | `1.35x0.84x0.80` | `0.371` | **CONFIRMED** |
| `desk_006` | desk | 17 | 100 | `0.97x0.36x0.70` | `0.127` | **CONFIRMED** |
| `desk_007` | desk | 43 | 3425 | `2.76x4.26x2.50` | `0.502` | **CONFIRMED** |
| `desk_009` | desk | 29 | 1805 | `0.98x1.33x0.80` | `0.124` | **CONFIRMED** |
| `desk_010` | desk | 43 | 590 | `1.52x1.99x0.82` | `0.178` | **CONFIRMED** |
| `desk_011` | desk | 33 | 585 | `1.10x1.90x0.78` | `0.279` | **CONFIRMED** |
| `desk_012` | desk | 128 | 2185 | `1.91x2.97x1.39` | `0.236` | **CONFIRMED** |
| `desk_014` | desk | 127 | 785 | `0.90x0.95x0.80` | `0.052` | **CONFIRMED** |
| `desk_015` | desk | 8 | 335 | `0.80x1.14x0.74` | `0.264` | **CONFIRMED** |
| `desk_016` | desk | 30 | 250 | `1.64x1.25x0.81` | `0.209` | **CONFIRMED** |
| `desk_020` | desk | 10 | 75 | `1.75x1.83x0.84` | `0.191` | **CONFIRMED** |
| `desk_022` | desk | 9 | 110 | `0.48x0.97x0.73` | `0.065` | **CONFIRMED** |
| `desk_023` | desk | 110 | 700 | `3.37x3.24x0.82` | `0.312` | **CONFIRMED** |
| `desk_027` | desk | 106 | 980 | `2.18x1.78x0.87` | `0.226` | **CONFIRMED** |
| `desk_029` | desk | 58 | 650 | `0.87x0.79x0.81` | `0.139` | **CONFIRMED** |
| `desk_032` | desk | 71 | 1505 | `0.67x1.21x0.79` | `0.202` | **CONFIRMED** |
| `desk_034` | desk | 45 | 700 | `1.32x1.45x0.90` | `0.202` | **CONFIRMED** |
| `desk_035` | desk | 58 | 1340 | `1.37x2.39x1.12` | `0.245` | **CONFIRMED** |
| `handbag_002` | backpack | 56 | 810 | `0.59x0.41x0.44` | `0.094` | **CONFIRMED** |
| `keyboard_003` | keyboard | 17 | 1045 | `0.31x0.38x0.36` | `0.237` | **CONFIRMED** |
| `keyboard_009` | keyboard | 22 | 655 | `0.30x0.43x0.07` | `0.074` | **CONFIRMED** |
| `keyboard_015` | keyboard | 13 | 410 | `0.29x0.50x0.07` | `0.286` | **CONFIRMED** |
| `laptop_001` | laptop | 17 | 4645 | `0.56x0.67x0.40` | `0.170` | **CONFIRMED** |
| `laptop_002` | laptop | 103 | 5175 | `1.08x1.02x0.37` | `0.258` | **CONFIRMED** |
| `laptop_003` | laptop | 21 | 1625 | `0.65x0.73x0.34` | `0.104` | **CONFIRMED** |
| `laptop_006` | laptop | 8 | 750 | `0.42x0.45x0.09` | `0.177` | **CONFIRMED** |
| `laptop_012` | laptop | 10 | 420 | `0.04x0.10x0.13` | `0.173` | **CONFIRMED** |
| `laptop_015` | laptop | 25 | 275 | `0.81x0.71x0.33` | `0.317` | **CONFIRMED** |
| `laptop_022` | laptop | 31 | 195 | `0.45x0.64x0.10` | `0.072` | **CONFIRMED** |
| `laptop_023` | laptop | 14 | 270 | `0.34x0.16x0.06` | `0.061` | **CONFIRMED** |
| `laptop_024` | laptop | 18 | 255 | `0.74x1.11x0.19` | `0.241` | **CONFIRMED** |
| `laptop_035` | laptop | 17 | 130 | `0.22x0.07x0.02` | `0.029` | **CONFIRMED** |
| `laptop_041` | laptop | 19 | 915 | `0.47x0.56x0.11` | `0.105` | **CONFIRMED** |
| `laptop_046` | laptop | 11 | 615 | `0.50x0.49x0.14` | `0.254` | **CONFIRMED** |
| `laptop_048` | laptop | 43 | 220 | `0.39x0.53x0.12` | `0.118` | **CONFIRMED** |
| `laptop_049` | laptop | 18 | 495 | `0.69x0.70x0.38` | `0.171` | **CONFIRMED** |
| `laptop_053` | laptop | 17 | 5140 | `0.40x0.57x0.33` | `0.256` | **CONFIRMED** |
| `quadcopter drone_001` | quadcopter drone | 8 | 510 | `0.08x0.25x0.09` | `0.023` | **CONFIRMED** |
| `storage box_003` | storage box | 12 | 270 | `0.41x0.38x0.34` | `0.050` | **CONFIRMED** |
| `storage box_004` | storage box | 14 | 255 | `2.94x3.92x2.85` | `0.183` | **CONFIRMED** |
| `storage box_007` | storage box | 8 | 805 | `0.98x0.63x0.44` | `0.264` | **CONFIRMED** |
| `storage box_010` | storage box | 53 | 555 | `0.45x0.59x0.50` | `0.071` | **CONFIRMED** |
| `storage box_011` | storage box | 30 | 310 | `0.25x0.21x0.24` | `0.060` | **CONFIRMED** |
| `storage box_013` | storage box | 9 | 40 | `0.32x0.37x0.08` | `0.080` | **CONFIRMED** |
| `backpack_002` | backpack | 3 | 65 | `1.12x0.49x0.44` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `backpack_003` | backpack | 1 | 0 | `0.83x0.67x0.58` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `ceiling fan_001` | ceiling fan | 5 | 20 | `1.13x0.81x0.72` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `ceiling fan_002` | ceiling fan | 5 | 30 | `0.54x0.69x0.47` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `ceiling fan_003` | ceiling fan | 2 | 830 | `0.36x0.39x0.12` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `ceiling fan_004` | ceiling fan | 7 | 35 | `0.45x0.42x0.12` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `ceiling fan_005` | ceiling fan | 2 | 20 | `0.02x0.01x0.03` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `chair_002` | chair | 4 | 4270 | `0.56x0.77x1.06` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `chair_003` | chair | 7 | 780 | `1.10x0.45x0.73` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `chair_010` | chair | 1 | 0 | `0.05x0.04x0.03` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_013` | chair | 1 | 0 | `0.15x0.15x0.23` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_016` | chair | 4 | 1110 | `0.48x0.96x0.25` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `chair_021` | chair | 1 | 0 | `0.30x0.17x0.37` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_024` | chair | 4 | 1175 | `0.73x1.05x0.86` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `chair_026` | chair | 1 | 0 | `0.81x0.65x0.33` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_027` | chair | 2 | 5 | `0.13x0.04x0.01` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `chair_028` | chair | 2 | 270 | `0.26x0.15x0.25` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `chair_029` | chair | 5 | 575 | `0.53x0.58x0.48` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `chair_034` | chair | 1 | 0 | `0.03x0.07x0.01` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_039` | chair | 7 | 335 | `1.47x0.74x0.78` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `chair_040` | chair | 6 | 25 | `0.11x0.27x0.30` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `chair_042` | chair | 1 | 0 | `0.36x0.85x0.11` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_043` | chair | 1 | 0 | `0.36x0.85x0.11` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_047` | chair | 1 | 0 | `0.17x0.15x0.22` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_048` | chair | 5 | 40 | `0.27x0.20x0.08` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `computer monitor_004` | computer monitor | 5 | 950 | `0.24x0.53x0.35` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `computer monitor_006` | computer monitor | 7 | 2180 | `0.37x0.62x0.38` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `computer monitor_007` | computer monitor | 1 | 0 | `0.33x0.35x0.33` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `computer monitor_008` | computer monitor | 1 | 0 | `0.20x0.26x0.25` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `computer monitor_009` | computer monitor | 1 | 0 | `0.48x0.81x0.10` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `computer monitor_010` | computer monitor | 2 | 5 | `0.60x0.74x0.22` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `computer monitor_011` | computer monitor | 1 | 0 | `0.14x0.44x0.33` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `computer mouse_001` | computer mouse | 1 | 0 | `0.40x0.57x0.04` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_013` | desk | 1 | 0 | `0.48x1.27x0.84` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_018` | desk | 5 | 90 | `1.28x0.94x0.56` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `desk_019` | desk | 5 | 70 | `1.87x1.78x0.76` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `desk_021` | desk | 3 | 50 | `0.43x0.78x0.73` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `desk_024` | desk | 2 | 10 | `0.32x0.60x0.27` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `desk_026` | desk | 2 | 785 | `0.67x0.53x0.36` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `desk_030` | desk | 5 | 35 | `0.60x0.26x0.48` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `desk_036` | desk | 1 | 0 | `0.05x0.22x0.30` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `door_001` | door | 1 | 0 | `0.88x0.25x0.88` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `door_002` | door | 1 | 0 | `0.95x0.31x0.75` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `handbag_001` | handbag | 1 | 0 | `0.13x0.61x0.22` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `handbag_003` | handbag | 1 | 0 | `0.01x0.01x0.03` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `handbag_004` | handbag | 1 | 0 | `0.14x0.12x0.10` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `handbag_005` | handbag | 3 | 1035 | `0.23x0.14x0.27` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `handbag_006` | handbag | 1 | 0 | `0.31x0.19x0.34` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `keyboard_001` | keyboard | 2 | 270 | `0.67x0.48x0.03` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `keyboard_002` | keyboard | 5 | 1210 | `0.16x0.12x0.03` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `keyboard_004` | keyboard | 2 | 70 | `0.27x0.29x0.05` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `keyboard_005` | keyboard | 2 | 10 | `0.14x0.36x0.07` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `keyboard_006` | keyboard | 6 | 2100 | `0.36x0.26x0.06` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `keyboard_008` | keyboard | 1 | 0 | `0.05x0.10x0.02` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `keyboard_010` | keyboard | 1 | 0 | `0.05x0.25x0.03` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `keyboard_011` | keyboard | 1 | 0 | `0.18x0.20x0.06` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `keyboard_012` | keyboard | 2 | 5 | `0.38x0.25x0.05` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `keyboard_013` | keyboard | 2 | 45 | `0.62x0.21x0.06` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `keyboard_014` | keyboard | 4 | 25 | `0.28x0.47x0.06` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `keyboard_016` | keyboard | 7 | 40 | `0.28x0.65x0.05` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `keyboard_017` | keyboard | 1 | 0 | `0.06x0.11x0.03` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `keyboard_018` | keyboard | 1 | 0 | `0.18x0.05x0.01` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `laptop_004` | laptop | 5 | 1330 | `0.42x0.50x0.31` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `laptop_005` | laptop | 6 | 5145 | `1.83x1.03x0.22` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `laptop_010` | laptop | 5 | 70 | `0.95x0.52x0.26` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `laptop_011` | laptop | 2 | 1170 | `0.40x0.59x0.30` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `laptop_013` | laptop | 6 | 1655 | `0.42x0.69x0.37` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `laptop_014` | laptop | 5 | 1620 | `0.66x0.42x0.22` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `laptop_016` | laptop | 4 | 565 | `0.18x0.17x0.13` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `laptop_017` | laptop | 6 | 95 | `0.49x0.49x0.36` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `laptop_018` | laptop | 3 | 90 | `0.47x0.49x0.31` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `laptop_019` | laptop | 2 | 20 | `0.10x0.07x0.12` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `laptop_020` | laptop | 2 | 5 | `0.00x0.01x0.00` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `laptop_021` | laptop | 7 | 120 | `0.07x0.21x0.02` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `laptop_025` | laptop | 5 | 150 | `0.55x0.78x0.22` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `laptop_026` | laptop | 1 | 0 | `0.37x0.50x0.16` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `laptop_027` | laptop | 2 | 5 | `0.33x0.57x0.22` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `laptop_028` | laptop | 4 | 30 | `0.68x0.68x0.20` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `laptop_029` | laptop | 7 | 70 | `0.19x0.07x0.03` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `laptop_030` | laptop | 3 | 10 | `0.27x0.29x0.17` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `laptop_031` | laptop | 1 | 0 | `0.17x0.45x0.13` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `laptop_032` | laptop | 2 | 10 | `0.42x1.74x0.33` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `laptop_033` | laptop | 1 | 0 | `0.62x0.42x0.20` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `laptop_034` | laptop | 1 | 0 | `0.16x0.07x0.07` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `laptop_036` | laptop | 3 | 50 | `0.17x0.42x0.13` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `laptop_037` | laptop | 1 | 0 | `0.49x0.51x0.18` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `laptop_038` | laptop | 1 | 0 | `0.28x0.17x0.03` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `laptop_039` | laptop | 6 | 405 | `0.97x0.77x0.29` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `laptop_040` | laptop | 3 | 65 | `0.45x0.45x0.27` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `laptop_042` | laptop | 1 | 0 | `0.38x0.42x0.27` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `laptop_043` | laptop | 1 | 0 | `0.42x0.45x0.27` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `laptop_044` | laptop | 2 | 10 | `0.31x0.70x0.24` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `laptop_045` | laptop | 1 | 0 | `0.16x0.19x0.07` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `laptop_047` | laptop | 5 | 50 | `0.19x0.11x0.04` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `laptop_050` | laptop | 2 | 25 | `0.62x0.66x0.09` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `laptop_051` | laptop | 3 | 70 | `0.83x0.94x0.17` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `laptop_052` | laptop | 1 | 0 | `0.43x1.80x0.36` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `pedestal fan_001` | pedestal fan | 2 | 35 | `0.44x0.54x0.28` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `person_001` | person | 1 | 0 | `0.00x0.00x0.02` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_002` | person | 1 | 0 | `0.05x0.12x0.15` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_003` | person | 16 | 75 | `0.89x0.82x1.04` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_004` | person | 7 | 1950 | `0.67x0.60x0.68` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_005` | person | 53 | 1865 | `2.44x0.88x0.93` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_006` | person | 36 | 1760 | `1.25x1.00x0.73` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_007` | person | 1 | 0 | `0.09x0.20x0.03` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_008` | person | 40 | 1460 | `1.49x1.76x1.72` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_009` | person | 34 | 1925 | `0.33x0.62x0.39` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_010` | person | 8 | 1155 | `0.38x0.49x0.12` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_011` | person | 8 | 35 | `1.09x1.14x0.79` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_012` | person | 11 | 210 | `0.80x0.95x0.57` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_013` | person | 3 | 10 | `0.85x0.44x0.82` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_014` | person | 18 | 555 | `0.83x0.75x0.64` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_015` | person | 5 | 20 | `0.91x0.77x0.62` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_016` | person | 11 | 220 | `1.16x0.90x0.80` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_017` | person | 17 | 220 | `1.15x0.75x0.73` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_018` | person | 27 | 540 | `2.01x1.42x1.13` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_019` | person | 47 | 445 | `1.84x0.23x0.33` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_020` | person | 27 | 590 | `0.57x0.41x0.37` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_021` | person | 11 | 490 | `1.82x0.68x1.10` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_022` | person | 19 | 335 | `1.75x0.75x0.43` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_023` | person | 7 | 360 | `1.34x0.84x1.06` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_024` | person | 54 | 895 | `0.76x0.84x0.50` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_025` | person | 4 | 840 | `0.28x0.22x0.05` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_026` | person | 4 | 35 | `0.72x0.76x0.10` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_027` | person | 28 | 405 | `1.54x1.82x0.43` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_028` | person | 36 | 260 | `1.73x1.87x0.73` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_029` | person | 2 | 5 | `0.10x0.11x0.23` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_030` | person | 52 | 665 | `0.67x0.82x0.50` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_031` | person | 1 | 0 | `0.23x0.13x0.11` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `projector_001` | projector | 6 | 80 | `0.50x0.43x0.07` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `projector_002` | projector | 5 | 35 | `0.08x0.20x0.04` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `projector_003` | projector | 1 | 0 | `0.17x0.11x0.02` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `projector_004` | projector | 3 | 265 | `0.15x0.17x0.05` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `projector_005` | projector | 1 | 0 | `0.09x0.46x0.02` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `quadcopter drone_002` | quadcopter drone | 1 | 0 | `0.01x0.01x0.03` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `robot_001` | robot | 2 | 40 | `0.82x0.52x0.25` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `storage box_001` | storage box | 2 | 3910 | `1.89x3.92x1.97` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `storage box_002` | storage box | 1 | 0 | `0.26x0.18x0.12` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `storage box_005` | storage box | 2 | 10 | `0.03x0.10x0.11` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `storage box_006` | storage box | 1 | 0 | `1.96x3.18x1.63` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `storage box_009` | storage box | 3 | 235 | `0.49x0.63x0.56` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `toolbox_001` | toolbox | 1 | 0 | `0.55x0.72x0.24` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `whiteboard_001` | whiteboard | 3 | 25 | `0.09x0.22x0.30` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `whiteboard_002` | whiteboard | 2 | 5 | `0.12x0.15x0.26` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `whiteboard_003` | whiteboard | 1 | 0 | `1.01x0.76x0.44` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |

---

## 5. Pairwise Clearances
Pairwise clearances computed with derived measurement uncertainty:
$\sigma_{AB} = \sqrt{\sigma_A^2 + \sigma_B^2 + 0.015^2}$
