# Multi-Session Dynamic Scene Graph Fusion Audit Report

**Reference Session 1**: `semantic_mapping_my_desk_recording_01`  
**Aligned Session 2**: `semantic_mapping_my_desk_recording_02`  
**Timestamp**: 2026-10-12 18:57:56  
**Output Directory**: `/mnt/c/Users/Yash Bhardwaj/Desktop/perception/outputs/semantic_mapping_fused_desk_run`  

---

## 1. Multi-Session Registration & Calibration
- **Anchor Alignment Algorithm**: Distance-Preserving RANSAC SE(2) Rigid Procrustes (Floor REP-103)
- **Rotation Angle**: `-16.13°` (verified physically consistent)
- **Translation Offset**: `[-0.081, 0.093] m`
- **Matched Anchor Objects**: `10` verified anchor pairs

---

## 2. Graph Expansion & Cumulative Entity Metrics

| Metric | Session 1 Alone | Session 2 Alone | Fused Scene Graph | Net Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Confirmed 3D Entities** | `41` | `69` | **`78`** | **+37 unique entities** |
| **Multi-View Enriched Entities** | - | - | **`32`** | Seamless 360° deduplication |
| **3D Point Cloud Vertices** | `701984` | `960160` | **`1662144`** | **+960160 points** |
| **Occupancy Unknown Space** | `63.3%` | `55.9%` | **`72.44%`** | **Significant blind spot reduction** |

---

## 3. Dynamic Change Detection Across Sessions
- **Objects Flagged as Displaced**: `3`
- **Significant Temporal Events**:
- **chair (chair_003)**: Displaced by **`0.39 m`** between session 1 and 2.
- **chair (chair_002)**: Displaced by **`0.56 m`** between session 1 and 2.
- **storage box (storage box_007)**: Displaced by **`0.40 m`** between session 1 and 2.

---

## 4. Deliverables Ready
- **Fused Dynamic Scene Graph**: `semantic_map.json`
- **Unified Nav2 Occupancy Grid**: `map.pgm` & `map.yaml`
- **Unified Nav2 Navigation Costmap**: `navigation_costmap.pgm` & `navigation_costmap.yaml`
- **Dense 3D Object Point Cloud**: `world_objects.ply` (25.4 MB)
- **High-Res Semantic Floor Plan**: `visualizations/semantic_floor_plan.png`
- **4-Panel Master Dashboard**: `visualizations/mapping_visualization.png`
