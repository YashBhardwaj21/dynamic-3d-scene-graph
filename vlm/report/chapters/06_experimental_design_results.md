# Chapter 6: Experimental Design and Results

## 6.1 Evaluation Populations & Protocol
To ensure rigorous, unbiased evaluation without target leakage, experiments are reported across dual standardized views:
- **View A (Matched Image-Eligible Population, N = 15)**: Instances possessing verified multi-view crop assets, stratified across all three taxonomy classes (3 IMMOVABLE, 9 MOVABLE, 3 MOBILE).
- **View B (Full System Population, N = 15)**: Evaluates complete system deployability, including missing visual inputs and safe abstentions.

## 6.2 Upstream Tracking Fragmentation & Tracklet Consolidation
The upstream ByteTrack bounding box tracker generates 1,970 unique tracks across the two sessions. Of these, 1,624 tracks (82.44%) terminate within 4 or fewer frames.
- **Cluster Consolidation Impact**: Intra-session consolidation agglomerates co-located tracks (< 0.35m threshold), reducing cluster counts by 8.82% in Session 1 and 12.05% in Session 2.
- **Cross-Session Recall Reality**: Against human identity annotations, consolidation maintains 7 of 30 recovered physical entities (23.08% static instance recall, 58.33% candidate precision, 5 false merges). The remaining unrecovered entities stem from severe track ID re-indexing in the front-end tracker.

## 6.3 Controlled Relocation Precision
On the three physical tape-measured controlled relocation events, the depth-weighted MAD spatial outlier rejection achieves:
- **Chair 002 (Caster chair)**: Measured 0.4889 m vs GT 0.5630 m (Error: 0.0741 m, 13.16%).
- **Chair 003 (Office chair)**: Measured 0.3512 m vs GT 0.3860 m (Error: 0.0348 m, 9.02%).
- **Storage Box 007 (Floor container)**: Measured 0.5729 m vs GT 0.3960 m (Error: 0.1769 m, 44.66%).
- **Aggregate**: MAE = **9.53 cm** (0.0953 m), Mean Relative Error = **22.28%** (strictly meeting the $\le 30.0\%$ acceptance criteria).
- **Static Stability**: 100.0% (0.0% false motion on stationary landmarks).

## 6.4 Stage 6 Selective Router Ablation Results
Table 6.3 and Figures 6.2–6.3 summarize the decoupled policy ablation on View A and View B:
- **R0 (Semantic Prior)**: 85.2% Balanced Accuracy, 0.0% queries.
- **R1 (Geometry Only)**: 74.1% Balanced Accuracy, 0.0% queries.
- **R2 (Non-Estimator-Aware)**: 88.9% Balanced Accuracy, 40.0% query rate (6 queries).
- **R3 (Estimator-Aware - Proposed)**: 100.0% Balanced Accuracy, 20.0% query rate (3 queries). Avoids 73.3%–80.0% of VLM queries.
- **R4 (All-Eligible VLM)**: 77.8% Balanced Accuracy, 100.0% query rate (15 queries). Suffers from linguistic mobility bias on architectural fixtures.
- **Random Query Baseline (1,000 seeds at k=3)**: 95.4% $\pm$ 6.2% Balanced Accuracy (CI [85.2%, 100.0%]).
- **R2 vs R3 Empirical Finding**: On the balanced manifest, R2 and R3 achieve high accuracy; R3's critical benefit is computational: it avoids 3 queries on background clutter lacking estimator relevance ($I_{\text{estimator}} = 0$).

## 6.5 Downstream SLAM Landmark Outcomes (Policy Decoupled)
Evaluated across decoupled scene graphs (R0 through R4) over identical observations:
- **R0 (Geometry-Only)**: 100.0% Retention (45/45), 0.0% False Invalidation (0/45), 25.0% Relocation Invalidation (1/4).
- **R1 (Heuristic Disparity Gate)**: 93.3% Retention (42/45), 6.7% False Invalidation (3/45), 50.0% Relocation Invalidation (2/4).
- **R2 (Semantic Uncertainty)**: 100.0% Retention (45/45), 0.0% False Invalidation (0/45), 25.0% Relocation Invalidation (1/4).
- **R3 (Estimator-Aware - Proposed)**: 91.1% Retention (41/45), 8.9% False Invalidation (4/45), 25.0% Relocation Invalidation (1/4).
- **Pose History Maintenance**: Preserves 15 history nodes across 3 physical relocated objects, preventing phantom centroid averaging.
