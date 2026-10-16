# Stage 5 Inexpensive Pre-Router Evidence Pipeline Evaluation

**Evaluation Timestamp:** `2026-10-16 04:12:07 UTC`  
**Total Records Evaluated:** `7,956`  
**Overall Acceptance Status:** `REPAIRS PENDING ⚠️`

---

## 1. Acceptance Gates Breakdown

| Gate Metric | Evaluated Value | Benchmark Threshold | Status | Diagnostic Reason |
|---|:---:|:---:|:---:|---|
| **Depth Validity Rate** | `100.0%` | `>= 95.0%` | **PASS ✅** | Depth validity is 100.0% (required >= 95.0%) |
| **False Motion Rate** | `0.0%` | `<= 2.0%` | **PASS ✅** | False motion rate on static structures is 0.00% (required <= 2.0%) |
| **Semantic Macro-F1** | `100.0%` | `>= 90.0%` | **PASS ✅** | Semantic Macro-F1 across 3/3 classes is 100.0% |
| **Controlled Displacement Error** | `0.0953m` | `<= 0.15m` | **PASS ✅** | Mean controlled relocation error is 9.5 cm (threshold 15 cm) |
| **Controlled Relative Error** | `22.28%` | `<= 30.0%` | **PASS ✅** | Mean relative displacement error is 22.3% (threshold 30%) |
| **Track Fragmentation Rate** | `72.7797833935018%` | `<= 25.0%` | **FAIL ❌** | Short-lifespan track fragmentation is 72.8% |

---

## 2. Geometric Consistency & Depth Uncertainty

- **Depth Validity Rate:** `100.0%` (Target: >= 95.0%)
- **Mean Depth Uncertainty (1-sigma):** `1.14 cm`
- **Point-to-BBox Distance (Median):** `3.21 px` (Evaluated on MapPoints outside box)
- **Point-to-BBox Inlier Ratio (Mean):** `70.94%`
- **Feature Reprojection Residual:** `None` (UNAVAILABLE_NO_KEYPOINT_CORRESPONDENCES)
- **Provenance Distribution:**
  - `MEASURED_FULL`: 7,666 records (96.4%)
  - `DERIVED_DEPTH_ONLY`: 290 records (3.6%)

---

## 3. Observed Motion & Controlled Relocations

| Physical Target | S1 Track | S2 Track | Ground Truth | Estimated Disp | Absolute Error | Relative Error | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| chair_002 (caster chair relocation) | `chair_002` | `chair_001` | `0.563 m` | `0.489 m` | `7.4 cm` | `13.2%` | **VERIFIED ✅** |
| chair_003 (office chair relocation) | `chair_003` | `chair_006` | `0.386 m` | `0.351 m` | `3.5 cm` | `9.0%` | **VERIFIED ✅** |
| box_007 (floor container relocation) | `storage box_007` | `storage box_013` | `0.396 m` | `0.573 m` | `17.7 cm` | `44.7%` | **VERIFIED ✅** |

- **Mean Controlled Displacement Error:** `9.53 cm`
- **False Motion Rate (Static Structures):** `0.0%` (Accuracy: 100.0%)

---

## 4. Semantic Evidence & Macro-F1

- **Evaluation Scope:** Conditional evaluation on matched taxonomy classes only.
- **Matched Support Observations:** `4,421` / `7,956` (`55.57%` evaluation coverage)
- **Unmapped / Excluded Observations:** `3,535` (`44.4%` out-of-vocabulary)
- **Matched-Support Macro-F1:** `100.00%` (3/3 classes evaluated)
- **Unknown Prediction Rate:** `0.0%`
- **Mean Shannon Entropy:** `0.0199`
- **Mean Semantic Purity:** `99.57%`

> [!WARNING]
> **Evaluation Coverage Notice:** The 100.00% Macro-F1 score is conditioned strictly on the 4,421 observations whose raw detector labels match the 3 evaluated taxonomy classes. 3,535 records (44.4%) were excluded from evaluation as unmapped. This score must NOT be cited as whole-pipeline classification accuracy.

| Class | Precision | Recall | F1-Score | Support | Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **IMMOVABLE** | `1.0000` | `1.0000` | `1.0000` | `13` | `EVALUATED` |
| **MOVABLE** | `1.0000` | `1.0000` | `1.0000` | `3566` | `EVALUATED` |
| **MOBILE** | `1.0000` | `1.0000` | `1.0000` | `842` | `EVALUATED` |

---

## 5. Sample Verified Evidence Records

| Entity ID | Category | Prior Mobility | Motion State | World Disp (m) | Depth (m) +/- Sigma | Point-Box Dist (px) | Decision |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `desk_001` | `desk` | `MOVABLE` | `STATIC` | `0.111` | `1.65 +/- 0.6cm` | `3.8 px` | `RESOLVE_CHEAP` |
| `chair_002` | `chair` | `MOVABLE` | `STATIC` | `0.037` | `1.29 +/- 0.4cm` | `3.0 px` | `RESOLVE_CHEAP` |
| `chair_003` | `chair` | `MOVABLE` | `DISPLACED` | `0.499` | `0.75 +/- 0.1cm` | `403.6 px` | `ESCALATE` |
| `door_001` | `door` | `IMMOVABLE` | `TRANSIENT` | `0.040` | `0.93 +/- 0.2cm` | `11.9 px` | `RESOLVE_CHEAP` |

---
*Report generated automatically by `vlm/scripts/evaluate_stage5_evidence.py`.*