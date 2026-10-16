# Research Artifact Plan: Reproducible Package & Technical Evaluation

## 1. Audit Findings & Baseline State

### 1.1 Environment & Git Lineage
- **Repository:** https://github.com/YashBhardwaj21/dynamic-3d-scene-graph
- **Current Branch:** `main`
- **Starting Commit SHA:** `4129e2efd7bbd390daaead8c95b797a67539bfcc`
- **Execution Runtime:** Python 3.13.7 (Win32), PyMuPDF 1.25.x, python-docx 1.1.2, Matplotlib 3.10.x, Pandas 2.2.x, Jinja2 3.1.x.
- **Headless Document Rendering Engine:** Microsoft Edge Headless (`msedge.exe --headless --disable-gpu --print-to-pdf`).

### 1.2 Authoritative Source Input Artifacts & Provenance Hashes
| Artifact | Size (Bytes) | SHA-256 Hash | Role in Pipeline |
| :--- | :---: | :---: | :--- |
| `vlm/outputs/object_observations.jsonl` | 6,477,341 | `8f6bed9551e57c6320868512e4c29697d2942b1e4ee537d7071aa21238dfab2e` | 7,956 raw multi-session YOLO detector observations |
| `vlm/outputs/stage5_registration_audit.json` | 12,935 | `d6ce09f45caa4e6a5f1811fdcda6d0513c537af63f86b521813c6729fc06a6dc` | Authoritative RANSAC SE(2) cross-session registration |
| `vlm/benchmarks/cross_session_identity_annotations.json` | 39,673 | `3a88a30261de3c19fcb1d020add2b0b34bc800bed881342297efdaddf22c0c23` | Multi-session ground-truth physical instance correspondences |
| `vlm/benchmarks/router_evaluation_manifest.json` | 7,647 | `b3d18685bb9c8cbf499fe3b02387c75f5c37ccc42e79a08b2e3c44daad103cb0` | 15-instance balanced evaluation manifest (View A) |
| `vlm/benchmarks/controlled_scene_events.json` | 3,723 | `3d7a59b8e2d5ea028482dbac87c0f29524b77057526efe55b4f242ae29708986` | Tape-measured physical relocation ground truth ($0.563, 0.386, 0.396\text{ m}$) |
| `vlm/benchmarks/canonical_benchmark_manifest.json` | 148,866 | `81f09e0c04070c748d2e9b0c392f1113539fe1f3089f05bd5590a7035ca66359` | 53 physical entities (View B all-instances census) |

### 1.3 Key Audit Diagnoses to Reconcile
1. **Tracklet Consolidation**:
   - Upstream front-end fragmentation audit confirms 1,970 unique tracks, of which 1,624 (82.44%) span $\le 4$ frames.
   - Consolidation reduces cluster counts (Session 1: $102 \to 93$, Session 2: $166 \to 146$), but benchmark recall against annotated instance naming remains 7 of 30 physical entities.
   - `run_full_experiment.py` must explicitly support a configurable raw vs consolidated comparison and report instance-level metrics truthfully without claiming 100% resolution.
2. **Semantic Coverage & Matched-Support F1**:
   - Out of 7,956 observations, 4,421 map to the 3 evaluated mobility classes (55.57% coverage), leaving 3,535 unmapped records.
   - Matched-support Macro-F1 is 100% on the mapped subset, but total pipeline coverage must be clearly reported alongside it to prevent misinterpretation.
3. **Downstream Scene Graph Evaluation**:
   - Decoupled scene graph generation (R0, R1, R2, R3, R4) must evaluate actual independent graphs.
   - Ground truth denominators must be strictly preserved; missing telemetry must be reported as `null` / unevaluated rather than fabricated.
   - Continuous trajectory ATE/RPE remains unevaluated (`null`) on `my_desk_sequence` due to absence of 6-DoF optical motion capture.
4. **Stage 6 Router Evidence & Latency**:
   - Cached replay latency ($0.000\text{ s}$) must remain distinct from live inference latency ($2.05\text{ s/call}$).
   - The random baseline query budget must be matched exactly ($k=4$).
   - Report actual per-policy accuracy and query counts directly from per-instance evaluation records.

---

## 2. Implementation & Evaluation Plan

### Phase 1: Pipeline & Evaluation Tooling Repairs
- **`vlm/src/cross_session_association.py`**:
  - Refine alias-aware ground-truth evaluation and transitive merge protection.
  - Implement explicit mapping from raw track IDs $\to$ consolidated cluster IDs $\to$ physical instance ground truth.
- **`vlm/src/stage5_adapter.py`**:
  - Rigorous provenance accounting for inlier counts, residuals, and displacement uncertainty. Mark missing or unverified fields as proxy/derived rather than hardcoded defaults.
- **`vlm/src/semantic_evidence.py`**:
  - Full taxonomy mapping, explicit unknown label accounting, and coverage calculation over all 7,956 records.
- **`vlm/src/router_policy.py`**:
  - Ensure estimator influence provenance is explicitly recorded (`MEASURED_BACKEND` vs `PROXY_SUPPORT`).
- **`vlm/scripts/run_full_experiment.py`**:
  - Integrate configurable raw vs consolidated association workflow, exporting decoupled scene graphs `scene_graph_r0.json` ... `scene_graph_r4.json`.
- **`vlm/scripts/evaluate_tracklet_consolidation.py`**:
  - Generate complete before vs after comparison with false merges, false splits, precision, and recall.
- **`vlm/scripts/run_stage6_ablation.py`**:
  - Consistent data-driven generation of R0–R4, random baseline ($k=4$), and oracle diagnostic.
- **`vlm/scripts/evaluate_policy_specific_downstream.py`**:
  - New script: execute decoupled downstream evaluation across R0–R4 on identical input observations.

### Phase 2: Reproducible Reporting Scripts
- **`vlm/scripts/generate_research_tables.py`**:
  - Generate Tables 1.1 to 7.2 in CSV, Markdown, and LaTeX from authoritative JSON results.
- **`vlm/scripts/generate_research_figures.py`**:
  - Generate Figures 3.1 to 7.1 in vector SVG and publication-grade 300 DPI PNG.
- **`vlm/scripts/build_project_report.py`**:
  - Compile the 8-chapter report and appendices A–M into `vlm/report/Project_Report.md`, `.docx`, and `.pdf`.
- **`vlm/scripts/validate_research_artifacts.py`**:
  - Automated integrity check verifying metric consistency across JSON, CSV, Markdown, and report prose.

---

## 3. Artifact Inventory & Output Directory Structure

The complete research package will be assembled under `results/` and mirrored in `vlm/outputs/research_package/`:
```text
results/
├── latest_run_manifest.json
├── artifact_manifest.json
├── metric_summary.csv
├── metric_summary.md
├── discrepancy_register.csv
├── tables/
│   ├── table_01_dataset_inventory.csv / .md / .tex
│   ├── table_02_stage5_metrics.csv / .md / .tex
│   ├── table_03_association_ablation.csv / .md / .tex
│   ├── table_04_router_ablation.csv / .md / .tex
│   ├── table_05_relocation_errors.csv / .md / .tex
│   ├── table_06_downstream_effects.csv / .md / .tex
│   └── ... remaining report tables
└── figures/
    ├── fig_3_1_system_architecture.svg / .png
    ├── fig_3_2_pipeline_data_flow.svg / .png
    ├── fig_4_1_router_decision_flow.svg / .png
    ├── fig_6_1_class_distributions.png
    ├── fig_6_2_confusion_matrices.png
    ├── fig_6_3_router_quality_cost.png
    ├── fig_6_4_track_lengths.png
    ├── fig_6_5_association_ablation.png
    ├── fig_6_6_relocation_errors.png
    ├── fig_6_7_landmark_outcomes.png
    └── fig_6_8_r2_r3_comparison.png
```

---

## 4. Acceptance Criteria
1. Every reported metric traces directly to an execution script and machine-readable output artifact.
2. All 121 existing unit tests pass without failure, plus new validation tests for metric consistency.
3. No fabricated numbers, no hidden fallbacks, no TUM trajectory numbers mixed into custom scene graph results.
4. Complete 8-chapter project report with Appendices A–M generated and built to Markdown, DOCX, and PDF.
