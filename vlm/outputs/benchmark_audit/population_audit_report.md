# Population Audit Report: Dynamic 3D Scene Graph VLM Subsystem

**Audited Repository Commit:** `7a71ddc7ff39bec53c4edf40685fee706e59e5db`  
**Audit Directory:** `vlm/outputs/benchmark_audit`  

---

## 1. Executive Summary & Population Reconciliations

This audit resolves the apparent population discrepancies across `canonical_benchmark_manifest.json` (53 entities), `movability_ground_truth.json` (52 objects), and multiview image caches:

| Evaluation Tier | Population Count | Composition | Usable Images | Estimator Evidence |
| :--- | :---: | :--- | :---: | :---: |
| **Total Physical Candidates** | **53** | All cataloged physical entities | 42 / 53 | 52 / 53 |
| **Frozen Pilot (Router)** | **15** | 15 balanced pilot instances | 14 / 15 | 15 / 15 |
| **Frozen Pilot (Visual)** | **14** | 14 image-eligible pilot instances | 14 / 14 | 14 / 14 |
| **Expanded Visual Population** | **42** | All audit-verified instances with decodable images | **42 / 42** | 41 / 42 |
| **Held-Out Furniture Subset** | **28** | 28 held-out furniture instances (all `MOVABLE`) | 28 / 28 | 28 / 28 |
| **Excluded from Visual Benchmark** | **11** | Physical entities lacking RGB crops in dataset | 0 / 11 | 11 / 11 |

---

## 2. Class Distribution Across Populations

### Expanded Visual Population (42 instances)
- `IMMOVABLE`: **3** (e.g., door, ceiling fan, whiteboard)
- `MOVABLE`: **37** (e.g., chairs, desks, backpacks, laptops, boxes, pedestal fan, monitor)
- `MOBILE`: **2** (e.g., human operator, quadcopter drone)

> [!WARNING]
> **Class Imbalance Disclosure**: The expanded visual population contains **37 MOVABLE**, **3 IMMOVABLE**, and **2 MOBILE** instances. The held-out furniture subset (28 instances) is 100% `MOVABLE`. Therefore, evaluation reports must present balanced accuracy and macro-F1 alongside overall accuracy to prevent prevalence bias.

---

## 3. Excluded Instances Detail (11 Entities)

The following 11 physical entities from the 52-object SLAM survey lack dedicated crop images in the dataset and are strictly excluded from visual VLM benchmarks:
- **`inst_desk_007_s2`** (desk): `MISSING_IMAGE_ASSETS`
- **`inst_desk_010_s2`** (desk): `MISSING_IMAGE_ASSETS`
- **`inst_handbag_001`** (handbag): `MISSING_IMAGE_ASSETS`
- **`inst_keyboard_001`** (keyboard): `MISSING_IMAGE_ASSETS`
- **`inst_laptop_002`** (laptop): `MISSING_IMAGE_ASSETS`
- **`inst_laptop_003`** (laptop): `MISSING_IMAGE_ASSETS`
- **`inst_laptop_004`** (laptop): `MISSING_IMAGE_ASSETS`
- **`inst_monitor_002`** (computer monitor): `MISSING_IMAGE_ASSETS`
- **`inst_rc_car_robot_001`** (robot): `MISSING_IMAGE_ASSETS`
- **`inst_wall_north_001`** (wall): `MISSING_IMAGE_ASSETS`
- **`inst_wall_west_001`** (wall): `MISSING_IMAGE_ASSETS`

---

## 4. Verification Check
- **Unique Physical Instance IDs**: Confirmed (53 unique canonical entities).
- **Image Decoding & Cryptographic Hashes**: All 42 visual instances decoded cleanly via PIL and verified with 64-character SHA-256 digests.
- **Physical vs. Observation Distinction**: Confirmed. 7,956 raw video frame observations map to 52 persistent object tracks, preventing repeated observation inflation.
