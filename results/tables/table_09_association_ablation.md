### Table 09 Association Ablation: Cross-Session Association Baseline Comparison Before vs After Tracklet Consolidation

| Baseline / Matcher | Condition | Matched Pairs | Annotated Pairs | True Positives | False Merges | Candidate Precision | Static Recall | Recovered Instances |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Nearest Centroid | Raw Clusters | 48 | 12 | 7 | 5 | 58.33% | 23.08% | 7 / 30 |
| Nearest Centroid | Consolidated | 48 | 12 | 7 | 5 | 58.33% | 23.08% | 7 / 30 |
| Hungarian Gated (0.45m) | Raw Clusters | 43 | 12 | 7 | 5 | 58.33% | 23.08% | 7 / 30 |
| Hungarian Gated (0.45m) | Consolidated | 43 | 12 | 7 | 5 | 58.33% | 23.08% | 7 / 30 |
| Uncertainty Symmetric | Raw Clusters | 33 | 11 | 7 | 4 | 63.64% | 23.08% | 7 / 30 |
| Uncertainty Symmetric | Consolidated | 33 | 11 | 7 | 5 | 58.33% | 23.08% | 7 / 30 |

*Consolidation merges intra-session fragmented tracklets but instance recall remains bounded at 7/30 due to upstream track ID naming disparity.*
