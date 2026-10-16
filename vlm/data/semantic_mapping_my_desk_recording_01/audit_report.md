# Offline Semantic Mapping Audit Report

**Sequence**: `my_desk_recording_01`
**Timestamp**: 2026-10-12 14:49:41
**Pipeline Mode**: `fresh`
**Output Path**: `/mnt/c/Users/Yash Bhardwaj/Desktop/perception/outputs/semantic_mapping_my_desk_recording_01`

---

## 1. Sequence Provenance & Coverage
- **Total RGB Frames**: `5207`
- **Total Depth Frames**: `5271`
- **Synchronized RGB-D Pairs**: `5108`
- **Sampling Policy**: `step` (step=5, fps=5.0)
- **Sampled Frames Processed**: `1022`
- **Input Coverage**: `20.01%` of all available sequence frames
- **Pose Alignment Coverage**: `100.0%` of sampled frames matched to SLAM poses

---

## 2. Floor Coordinate System & Transform
- **Frame Convention**: ROS REP-103 (`floor_map` frame: X=Forward, Y=Left, Z=Height above floor)
- **Fitted Floor Plane**:
  - Upward Normal: `[-0.0528, -0.9910, 0.1232]`
  - Plane Distance $d$: `0.4000 m`
  - Inliers: `23079/34823 (66.28%)`
  - Residual Mean/Std: `9.2 mm` / `8.1 mm`
- **Transform**: Full $4 \times 4$ matrix $T_{\text{floor\_world}}$ stored in `floor_plane.json`.

---

## 3. Dynamic Occupancy Grid & Obstacle Clearances
- **Origin**: `(-6.250, -8.800) m`
- **Dimensions**: `13.00 x 12.00 m` @ `0.05 m/cell` (260 x 240 cells)
- **Free Cells**: `9668 (15.49%)`
- **Occupied Cells**: `13225 (21.19%)`
- **Unknown Cells**: `39507 (63.31%)`
- **Inflated Navigation Occupied Cells**: `28303 (45.36%)`
- **Robot Radius / Safety Margin**: `0.18 m` / `0.05 m`

---

## 4. Semantic Objects Summary
- **Confirmed Static Entities**: `41`
- **Quarantined Candidates**: `97`

| Entity ID | Class | Observations | Frame Span | Dimensions (WxLxH m) | Spread $\sigma$ (m) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `backpack_002` | backpack | 72 | 2580 | `0.46x0.57x0.43` | `0.078` | **CONFIRMED** |
| `backpack_003` | backpack | 242 | 2790 | `0.51x0.63x0.51` | `0.060` | **CONFIRMED** |
| `ceiling fan_010` | ceiling fan | 10 | 45 | `0.51x0.62x0.51` | `0.127` | **CONFIRMED** |
| `chair_001` | chair | 12 | 3250 | `1.21x0.94x0.81` | `0.273` | **CONFIRMED** |
| `chair_002` | chair | 91 | 2755 | `1.57x1.41x0.86` | `0.230` | **CONFIRMED** |
| `chair_003` | chair | 182 | 3050 | `1.04x1.78x0.95` | `0.249` | **CONFIRMED** |
| `chair_004` | chair | 53 | 2815 | `1.32x2.04x1.04` | `0.349` | **CONFIRMED** |
| `chair_005` | chair | 212 | 3155 | `1.15x1.74x0.99` | `0.163` | **CONFIRMED** |
| `chair_007` | chair | 159 | 3230 | `0.98x1.20x0.83` | `0.211` | **CONFIRMED** |
| `chair_008` | chair | 87 | 3225 | `1.75x2.62x0.97` | `0.266` | **CONFIRMED** |
| `chair_009` | chair | 133 | 2855 | `1.20x1.53x0.81` | `0.160` | **CONFIRMED** |
| `chair_011` | chair | 109 | 2655 | `0.76x1.17x0.85` | `0.184` | **CONFIRMED** |
| `chair_017` | chair | 120 | 2600 | `1.84x0.92x0.95` | `0.158` | **CONFIRMED** |
| `chair_018` | chair | 20 | 2235 | `0.92x0.39x0.82` | `0.187` | **CONFIRMED** |
| `chair_019` | chair | 46 | 2470 | `1.08x1.16x0.90` | `0.221` | **CONFIRMED** |
| `chair_021` | chair | 12 | 215 | `1.43x2.24x0.69` | `0.225` | **CONFIRMED** |
| `chair_027` | chair | 122 | 2850 | `0.83x0.89x0.93` | `0.258` | **CONFIRMED** |
| `chair_030` | chair | 71 | 2530 | `0.67x0.95x0.82` | `0.169` | **CONFIRMED** |
| `chair_032` | chair | 21 | 155 | `1.43x1.22x0.60` | `0.221` | **CONFIRMED** |
| `chair_035` | chair | 11 | 55 | `0.65x1.06x0.81` | `0.056` | **CONFIRMED** |
| `chair_038` | chair | 35 | 220 | `0.99x0.91x0.83` | `0.200` | **CONFIRMED** |
| `computer monitor_001` | computer monitor | 11 | 2485 | `0.30x0.60x0.34` | `0.048` | **CONFIRMED** |
| `desk_001` | desk | 37 | 5010 | `1.35x1.13x0.82` | `0.292` | **CONFIRMED** |
| `desk_003` | desk | 28 | 2780 | `0.95x0.80x0.70` | `0.199` | **CONFIRMED** |
| `desk_004` | desk | 30 | 4525 | `1.10x0.91x0.81` | `0.141` | **CONFIRMED** |
| `desk_005` | desk | 79 | 2805 | `1.02x0.77x0.80` | `0.058` | **CONFIRMED** |
| `desk_008` | desk | 31 | 2535 | `1.00x1.40x0.85` | `0.256` | **CONFIRMED** |
| `desk_012` | desk | 12 | 2310 | `1.97x1.33x0.73` | `0.277` | **CONFIRMED** |
| `desk_014` | desk | 121 | 2535 | `2.63x3.28x1.21` | `0.225` | **CONFIRMED** |
| `desk_016` | desk | 30 | 2545 | `1.03x0.92x0.59` | `0.415` | **CONFIRMED** |
| `desk_017` | desk | 91 | 2555 | `2.35x2.43x0.88` | `0.196` | **CONFIRMED** |
| `desk_018` | desk | 89 | 2470 | `1.09x0.95x0.84` | `0.145` | **CONFIRMED** |
| `desk_023` | desk | 16 | 2015 | `1.01x1.20x0.76` | `0.090` | **CONFIRMED** |
| `desk_024` | desk | 28 | 2220 | `0.65x0.70x0.73` | `0.157` | **CONFIRMED** |
| `desk_025` | desk | 8 | 70 | `1.12x0.73x0.38` | `0.082` | **CONFIRMED** |
| `desk_031` | desk | 9 | 460 | `1.07x1.23x0.75` | `0.046` | **CONFIRMED** |
| `laptop_006` | laptop | 9 | 2280 | `0.41x2.21x0.44` | `0.305` | **CONFIRMED** |
| `pedestal fan_003` | pedestal fan | 8 | 180 | `0.35x0.27x0.35` | `0.049` | **CONFIRMED** |
| `storage box_001` | storage box | 18 | 2685 | `0.61x0.45x0.62` | `0.199` | **CONFIRMED** |
| `storage box_004` | storage box | 11 | 2270 | `0.23x0.24x0.22` | `0.062` | **CONFIRMED** |
| `storage box_007` | storage box | 11 | 480 | `0.61x0.32x0.52` | `0.050` | **CONFIRMED** |
| `backpack_001` | backpack | 7 | 50 | `0.28x0.21x0.36` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `backpack_004` | backpack | 1 | 0 | `0.18x0.18x0.16` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `backpack_005` | backpack | 1 | 0 | `0.14x0.14x0.16` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `backpack_006` | backpack | 1 | 0 | `0.01x0.00x0.05` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `backpack_007` | backpack | 1 | 0 | `0.06x0.11x0.17` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `ceiling fan_001` | ceiling fan | 1 | 0 | `0.48x0.72x0.27` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `ceiling fan_002` | ceiling fan | 2 | 5 | `0.05x0.04x0.05` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `ceiling fan_003` | ceiling fan | 6 | 2590 | `0.52x0.39x0.31` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `ceiling fan_004` | ceiling fan | 5 | 155 | `0.08x0.09x0.03` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `ceiling fan_005` | ceiling fan | 1 | 0 | `0.08x0.05x0.02` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `ceiling fan_006` | ceiling fan | 2 | 5 | `0.02x0.02x0.03` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `ceiling fan_007` | ceiling fan | 3 | 2145 | `0.02x0.02x0.03` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `ceiling fan_008` | ceiling fan | 1 | 0 | `0.47x0.41x0.15` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `ceiling fan_009` | ceiling fan | 1 | 0 | `0.18x0.11x0.06` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `ceiling fan_011` | ceiling fan | 1 | 0 | `0.20x0.24x0.11` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_015` | chair | 6 | 35 | `0.50x0.66x0.90` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `chair_016` | chair | 1 | 0 | `0.52x1.02x0.05` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_020` | chair | 4 | 665 | `0.73x2.27x0.92` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `chair_022` | chair | 1 | 0 | `0.34x0.43x0.10` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_023` | chair | 3 | 1940 | `0.26x0.13x0.26` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `chair_024` | chair | 1 | 0 | `0.15x0.09x0.11` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_025` | chair | 2 | 5 | `0.20x0.45x0.08` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `chair_026` | chair | 1 | 0 | `0.14x0.12x0.05` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_028` | chair | 4 | 85 | `0.70x0.84x0.61` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `chair_029` | chair | 1 | 0 | `0.14x0.34x0.19` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_031` | chair | 6 | 2000 | `0.40x0.39x0.46` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `chair_039` | chair | 3 | 30 | `1.72x1.01x0.86` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `chair_041` | chair | 7 | 190 | `0.90x0.99x0.87` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `chair_042` | chair | 1 | 0 | `0.83x0.42x0.43` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `computer monitor_002` | computer monitor | 1 | 0 | `0.29x0.34x0.18` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `computer monitor_003` | computer monitor | 6 | 100 | `0.28x0.41x0.33` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `computer monitor_004` | computer monitor | 2 | 15 | `0.87x0.67x0.36` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `desk_002` | desk | 1 | 0 | `0.40x0.24x0.17` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_006` | desk | 1 | 0 | `0.16x1.53x0.22` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_007` | desk | 1 | 0 | `0.14x0.57x0.24` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_009` | desk | 6 | 50 | `0.90x0.90x0.75` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `desk_010` | desk | 1 | 0 | `0.31x0.40x0.44` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_011` | desk | 2 | 5 | `0.74x0.62x0.73` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `desk_013` | desk | 2 | 10 | `1.81x0.97x0.73` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `desk_020` | desk | 3 | 1955 | `0.36x0.20x0.28` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `desk_021` | desk | 3 | 2095 | `0.26x0.31x0.16` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `desk_022` | desk | 5 | 25 | `0.93x1.54x0.44` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `desk_026` | desk | 1 | 0 | `0.29x0.19x0.09` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_027` | desk | 1 | 0 | `0.36x0.18x0.59` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_028` | desk | 2 | 75 | `0.02x0.02x0.03` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `desk_029` | desk | 1 | 0 | `0.03x0.03x0.04` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_030` | desk | 1 | 0 | `0.08x0.06x0.29` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `door_001` | door | 2 | 5 | `0.58x0.37x0.89` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `door_002` | door | 2 | 115 | `0.68x0.39x0.92` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `door_003` | door | 4 | 2050 | `0.94x0.57x0.90` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `door_004` | door | 1 | 0 | `2.19x2.38x1.96` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `door_005` | door | 6 | 205 | `3.69x2.43x3.06` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `handbag_001` | handbag | 5 | 2305 | `0.54x0.46x0.26` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `handbag_002` | handbag | 2 | 55 | `0.68x0.49x0.32` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `handbag_003` | handbag | 4 | 65 | `0.47x0.43x0.20` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `laptop_001` | laptop | 1 | 0 | `0.29x0.18x0.02` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `laptop_002` | laptop | 1 | 0 | `0.17x0.35x0.05` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `laptop_003` | laptop | 6 | 2380 | `0.28x0.47x0.31` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `laptop_004` | laptop | 4 | 2220 | `0.28x0.40x0.16` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `laptop_005` | laptop | 7 | 250 | `0.88x0.54x0.31` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (7 < 8)) |
| `laptop_007` | laptop | 3 | 2265 | `0.27x1.04x0.23` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `laptop_008` | laptop | 1 | 0 | `0.19x0.20x0.19` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `laptop_009` | laptop | 2 | 10 | `0.34x0.46x0.15` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `laptop_010` | laptop | 3 | 20 | `0.18x0.32x0.18` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `pedestal fan_001` | pedestal fan | 1 | 0 | `0.07x0.05x0.07` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `pedestal fan_002` | pedestal fan | 3 | 280 | `0.17x0.24x0.27` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `person_001` | person | 17 | 2340 | `0.64x0.22x0.56` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_002` | person | 6 | 145 | `0.68x0.25x0.35` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_003` | person | 32 | 2455 | `2.32x0.97x1.22` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_004` | person | 21 | 2430 | `0.43x0.38x0.25` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_005` | person | 68 | 2530 | `0.66x0.56x0.30` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_006` | person | 31 | 2460 | `0.57x0.54x0.36` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_007` | person | 10 | 2190 | `0.46x0.17x0.67` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_008` | person | 3 | 15 | `0.42x0.21x0.27` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_009` | person | 4 | 120 | `0.47x0.23x0.29` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_010` | person | 7 | 2145 | `0.56x0.25x0.38` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_011` | person | 3 | 10 | `0.49x0.73x0.92` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_012` | person | 6 | 25 | `1.36x1.15x1.40` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_013` | person | 5 | 20 | `1.32x0.96x1.68` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_014` | person | 5 | 20 | `1.18x1.26x1.67` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_015` | person | 10 | 150 | `0.39x0.38x0.36` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_016` | person | 5 | 125 | `0.18x0.15x0.09` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_017` | person | 3 | 130 | `0.35x0.26x0.22` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_018` | person | 1 | 0 | `0.08x0.08x0.02` | - | QUARANTINED (DYNAMIC_OBJECT) |
| `projector_001` | projector | 2 | 35 | `0.65x0.20x0.20` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `quadcopter drone_001` | quadcopter drone | 4 | 2265 | `0.54x0.50x0.26` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (4 < 8)) |
| `robot_001` | robot | 2 | 25 | `0.61x0.40x0.38` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `storage box_002` | storage box | 6 | 2345 | `0.25x0.43x0.39` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (6 < 8)) |
| `storage box_003` | storage box | 1 | 0 | `0.23x0.29x0.11` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `storage box_005` | storage box | 2 | 5 | `0.25x0.24x0.20` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `storage box_006` | storage box | 1 | 0 | `0.56x0.28x0.38` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `storage box_008` | storage box | 1 | 0 | `0.15x0.11x0.33` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `storage box_009` | storage box | 1 | 0 | `0.22x0.14x0.44` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `storage box_010` | storage box | 1 | 0 | `0.22x0.14x0.08` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `toolbox_001` | toolbox | 2 | 5 | `0.20x0.30x0.20` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `whiteboard_001` | whiteboard | 3 | 2505 | `1.81x2.16x0.79` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `whiteboard_002` | whiteboard | 1 | 0 | `0.30x0.44x0.25` | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |

---

## 5. Pairwise Clearances
Pairwise clearances computed with derived measurement uncertainty:
$\sigma_{AB} = \sqrt{\sigma_A^2 + \sigma_B^2 + 0.015^2}$
