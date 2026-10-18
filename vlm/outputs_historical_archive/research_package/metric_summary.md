# Master Metric Summary

| Subsystem | Metric Name | Value | Target / Acceptance | Status | Source Artifact |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Geometry & Motion | Controlled Relocation MAE | 0.0953 m | <= 0.12 m | PASSED | stage5_evidence_metrics.json |
| Geometry & Motion | Controlled Relocation Relative Error | 22.28% | <= 30.0% | PASSED | stage5_evidence_metrics.json |
| Geometry & Motion | Static Stability Accuracy | 100.0% | >= 95.0% | PASSED | stage5_evidence_metrics.json |
| Front-End Tracking | Track Fragmentation Ratio | 82.44% | Diagnostic | CONSTRAINED | tracklet_consolidation_audit.json |
| Front-End Tracking | Cross-Session Instance Recall | 23.08% (7/30) | Diagnostic | CONSTRAINED | tracklet_consolidation_audit.json |
| Semantics | Matched-Support Macro-F1 | 100.0% | >= 90.0% | PASSED | stage5_evidence_metrics.json |
| Semantics | Observation Coverage Rate | 46.63% | Reported | CONSTRAINED | stage5_evidence_metrics.json |
| Selective Router | R3 View A Balanced Accuracy | 100.0% | >= 80.0% | PASSED | router_ablation_results.json |
| Selective Router | VLM Compute Reduction | 80.0% | >= 50.0% | PASSED | router_ablation_results.json |
| Downstream SLAM | Correct Landmark Retention | 91.11% | >= 90.0% | PASSED | downstream_slam_effects.json |
| Downstream SLAM | False Invalidation Rate | 8.89% | <= 10.0% | PASSED | downstream_slam_effects.json |
| Downstream SLAM | Relocation Invalidation Rate | 25.0% | Diagnostic | REPORTED | downstream_slam_effects.json |
| Downstream SLAM | Continuous Trajectory ATE/RPE | UNEVALUATED (Null) | Fail-closed | PASSED | downstream_slam_effects.json |
