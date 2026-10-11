# Offline Semantic Mapping & Geometry Audit Report (Run 2)

**Dataset**: `d455_rgbd_20261010_123443`
**Coordinate Frame**: `floor_map (ROS REP-103: X=Forward, Y=Left, Z=Height, Z=0 on floor)`

---

## 1. Coordinate Provenance & Unified Transform
- **Frame Convention**: ROS REP-103 standard (`floor_map` frame).
  - Horizontal plane: $X$ (forward), $Y$ (left)
  - Vertical: $Z$ (height above floor, $Z=0$ is floor plane)
- **Fitted Floor Plane**:
  - Upward Normal: `[0.1697, -0.9835, -0.0631]`
  - Plane Offset $d$: `0.5167 m`
  - Inliers: `22246/25240 (88.14%)`
  - Plane Residual: Mean `5.0 mm`, Std `5.0 mm`
  - Camera Height at Origin: `0.517 m`
- **Unified Transform Matrix (`T_floor_world`)**:
  - Saved in `floor_plane.json`. Every camera pose, object point cloud, bounding box, footprint, and grid raycast is transformed through this single matrix.

---

## 2. Perception & Provenance Summary
- **Total Processed Frames**: `499` (sampled every 5 frames)
- **Raw Detections**: `2420`
- **Valid 3D Observations**: `2096`
- **Sequential Local Tracking**: Every detection and observation records its `local_track_id` and assigned `global_entity_id`.

---

## 3. Persistent Object Registry & Confirmation
- **Confirmed World Objects**: `17`
- **Quarantined Candidates**: `26` (filtered out by strict confirmation policy or dynamic flag)

| Entity ID | Class | Observations | Frame Span | Dimensions (WxLxH m) | Centroid Spread $\sigma$ (m) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `backpack_001` | backpack | 332 | 815 | `0.69x1.06x0.43` | `0.197` | **CONFIRMED** |
| `backpack_005` | backpack | 48 | 245 | `0.31x0.41x0.42` | `0.054` | **CONFIRMED** |
| `ceiling fan_002` | ceiling fan | 9 | 180 | `0.63x1.07x0.77` | `0.200` | **CONFIRMED** |
| `chair_001` | chair | 134 | 665 | `1.47x0.74x0.83` | `0.092` | **CONFIRMED** |
| `chair_002` | chair | 110 | 810 | `1.02x0.61x0.86` | `0.188` | **CONFIRMED** |
| `chair_004` | chair | 80 | 475 | `1.48x0.48x0.83` | `0.134` | **CONFIRMED** |
| `chair_011` | chair | 12 | 70 | `1.36x0.28x0.78` | `0.204` | **CONFIRMED** |
| `chair_012` | chair | 91 | 440 | `0.65x0.94x0.81` | `0.166` | **CONFIRMED** |
| `chair_013` | chair | 119 | 560 | `1.16x1.28x0.95` | `0.194` | **CONFIRMED** |
| `chair_016` | backpack | 118 | 615 | `0.49x0.74x0.70` | `0.131` | **CONFIRMED** |
| `chair_018` | chair | 68 | 585 | `0.50x1.04x0.95` | `0.217` | **CONFIRMED** |
| `desk_005` | desk | 11 | 105 | `1.33x1.05x0.97` | `0.180` | **CONFIRMED** |
| `desk_007` | desk | 93 | 380 | `2.66x1.59x1.25` | `0.326` | **CONFIRMED** |
| `door_001` | door | 115 | 720 | `1.27x1.84x2.33` | `0.256` | **CONFIRMED** |
| `handbag_001` | handbag | 162 | 720 | `0.37x0.65x0.43` | `0.117` | **CONFIRMED** |
| `handbag_003` | backpack | 122 | 655 | `0.44x1.41x0.41` | `0.048` | **CONFIRMED** |
| `laptop_001` | laptop | 62 | 750 | `0.67x0.33x0.36` | `0.121` | **CONFIRMED** |
| `backpack_004` | backpack | 1 | 0 | - | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `ceiling fan_001` | ceiling fan | 5 | 135 | - | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (5 < 8)) |
| `chair_003` | desk | 63 | 550 | - | - | QUARANTINED (HIGH_CENTROID_VARIANCE (0.330m > 0.25m)) |
| `chair_005` | chair | 53 | 430 | - | - | QUARANTINED (HIGH_CENTROID_VARIANCE (0.256m > 0.25m)) |
| `chair_006` | chair | 1 | 0 | - | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `chair_007` | chair | 15 | 360 | - | - | QUARANTINED (HIGH_CENTROID_VARIANCE (0.267m > 0.25m)) |
| `chair_008` | chair | 21 | 245 | - | - | QUARANTINED (IMPLAUSIBLE_DIMENSIONS (0.72x0.23x0.68m < 0.28x0.28x0.35m)) |
| `chair_009` | chair | 2 | 10 | - | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `desk_001` | desk | 24 | 665 | - | - | QUARANTINED (HIGH_CENTROID_VARIANCE (0.303m > 0.25m)) |
| `desk_002` | desk | 3 | 10 | - | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (3 < 8)) |
| `desk_003` | desk | 55 | 350 | - | - | QUARANTINED (HIGH_CENTROID_VARIANCE (0.251m > 0.25m)) |
| `desk_004` | desk | 1 | 0 | - | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_006` | desk | 17 | 350 | - | - | QUARANTINED (HIGH_CENTROID_VARIANCE (0.312m > 0.25m)) |
| `desk_008` | desk | 1 | 0 | - | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |
| `desk_009` | desk | 31 | 435 | - | - | QUARANTINED (IMPLAUSIBLE_DIMENSIONS (0.23x1.11x0.71m < 0.40x0.40x0.35m)) |
| `door_002` | door | 2 | 65 | - | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (2 < 8)) |
| `person_001` | person | 6 | 25 | - | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_002` | person | 16 | 100 | - | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_003` | person | 10 | 75 | - | - | QUARANTINED (DYNAMIC_OBJECT) |
| `person_004` | person | 3 | 75 | - | - | QUARANTINED (DYNAMIC_OBJECT) |
| `storage box_001` | door | 24 | 295 | - | - | QUARANTINED (IMPLAUSIBLE_DIMENSIONS (0.40x0.28x0.53m < 0.35x0.10x0.70m)) |
| `storage box_002` | storage box | 12 | 205 | - | - | QUARANTINED (IMPLAUSIBLE_DIMENSIONS (0.07x0.24x0.10m < 0.20x0.20x0.15m)) |
| `storage box_003` | laptop | 10 | 140 | - | - | QUARANTINED (HIGH_CENTROID_VARIANCE (0.302m > 0.25m)) |
| `storage box_004` | storage box | 8 | 310 | - | - | QUARANTINED (HIGH_CENTROID_VARIANCE (0.266m > 0.25m)) |
| `storage box_006` | storage box | 25 | 305 | - | - | QUARANTINED (IMPLAUSIBLE_DIMENSIONS (0.13x0.33x0.74m < 0.20x0.20x0.15m)) |
| `storage box_007` | storage box | 1 | 0 | - | - | QUARANTINED (INSUFFICIENT_OBSERVATIONS (1 < 8)) |

---

## 4. Occupancy Grid Statistics
- **Total Cells**: `24928` (`7.6 x 8.2 m` @ `0.05 m`)
- **Free Cells**: `6054 (24.29%)`
- **Occupied Obstacle Cells**: `4282 (17.18%)`
- **Unknown Cells**: `14592 (58.54%)`
- **Dynamic Objects**: Persons were suppressed from the static occupancy grid.

---

## 5. Pairwise Distances & Derived Uncertainties
- Pairwise distances between all confirmed static objects computed with **measurement-derived uncertainty** $\sigma = \sqrt{\sigma_A^2 + \sigma_B^2 + \sigma_{\text{depth}}^2}$.
