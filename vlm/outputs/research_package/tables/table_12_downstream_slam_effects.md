### Table 12 Downstream Slam Effects: Policy-Specific Downstream Scene Graph Landmark Reliability Comparison

| Policy | Policy Description | Total Nodes | Landmark Retention | False Invalidation | Relocation Invalidation | Pose History Nodes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| R0 | Geometry-Only Prior (Zero VLM queries) | 194 | 100.0% (45/45) | 0.0% (0/45) | 25.0% (1/4) | 0 |
| R1 | Heuristic Geometric Disparity Gate (>= 0.25m) | 194 | 93.3% (42/45) | 6.7% (3/45) | 50.0% (2/4) | 11 |
| R2 | Semantic Uncertainty-Only (Non-estimator-aware) | 195 | 100.0% (45/45) | 0.0% (0/45) | 25.0% (1/4) | 0 |
| R3 | Estimator-Aware Selective Router (Proposed) | 195 | 91.1% (41/45) | 8.9% (4/45) | 25.0% (1/4) | 12 |
| R4 | All-Eligible VLM Reference Policy | 195 | 91.1% (41/45) | 8.9% (4/45) | 25.0% (1/4) | 12 |

*Evaluated over identical input observations. Trajectory ATE/RPE fail-closed as UNEVALUATED on custom room.*
