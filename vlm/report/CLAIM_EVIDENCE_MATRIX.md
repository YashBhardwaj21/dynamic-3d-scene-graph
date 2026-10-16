# Claim-Evidence Traceability Matrix

This matrix connects every principal research claim and assertion to its exact source script, machine-readable artifact, output fields, evaluation population, and verified status.

| Claim Identifier | Research Claim | Source Script | Source Artifact | Output Fields / Metrics | Population | Empirical Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CLAIM-01** | Standalone VLM suffers from linguistic mobility bias on architectural fixtures | `run_stage6_ablation.py` | `stage6/router_ablation_results.json` | `r4_all_eligible_vlm.view_a_metrics.quality.per_class_recall.IMMOVABLE` (= 0.0%) | View A (N=15) | **VERIFIED** |
| **CLAIM-02** | Selective router (R3) reduces VLM compute by >= 70% | `run_stage6_ablation.py` | `stage6/router_ablation_results.json` | `relative_compute_reduction_pct` (= 73.3%-80.0%) | View A (N=15) | **VERIFIED** |
| **CLAIM-03** | Estimator awareness gates compute on clutter without accuracy loss vs R2 | `run_stage6_ablation.py` | `stage6/router_ablation_results.json` | `r2.balanced_accuracy` vs `r3.balanced_accuracy` (equal on test set) | View A & View B | **VERIFIED (COMPUTE GATE)** |
| **CLAIM-04** | Depth-weighted MAD centroid filtering reduces relocation error to <= 30% | `evaluate_stage5_evidence.py` | `stage5_evidence_metrics.json` | `mean_relative_displacement_error_pct` (= 22.28%), `mean_displacement_error_m` (= 0.0953m) | 3 Controlled Relocations | **VERIFIED** |
| **CLAIM-05** | Tracklet consolidation reduces cluster count but recall remains bounded | `evaluate_tracklet_consolidation.py` | `tracklet_consolidation_audit.json` | `reduction_ratio_pct` (8.8%-12.1%), `recovered_instances_count` (= 7/30) | 1,970 tracks | **VERIFIED (NEGATIVE/CONSTRAINED)** |
| **CLAIM-06** | Stage 5 semantic matched-support macro-F1 is 100% on 55.57% coverage | `evaluate_stage5_evidence.py` | `stage5_evidence_metrics.json` | `matched_support_macro_f1` (= 1.0), `evaluation_coverage_pct` (= 55.57%) | 7,956 observations | **VERIFIED** |
| **CLAIM-07** | Selective routing preserves stationary landmarks and invalidates relocations | `evaluate_policy_specific_downstream.py` | `downstream_slam_effects.json` | `correct_retention_rate_pct` (= 91.1%), `false_invalidation_rate_pct` (= 8.9%) | 195 scene graph nodes | **VERIFIED** |
| **CLAIM-08** | Continuous trajectory ATE/RPE drift is reduced on my_desk_sequence | N/A (No MoCap) | `downstream_slam_effects.json` | `ate_rmse_m` (= null), `trajectory_status` (= UNEVALUATED) | `my_desk_sequence` | **UNEVALUATED (FAIL-CLOSED)** |
