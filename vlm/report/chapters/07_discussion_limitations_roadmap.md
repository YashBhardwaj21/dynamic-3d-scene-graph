# Chapter 7: Discussion, Limitations, and Roadmap

## 7.1 Interpretation of Verified Findings
1. **Linguistic Mobility Bias in Foundation VLMs**: Standalone zero-shot prompting on architectural structures (`door`, `ceiling fan`) frequently misclassifies them as `MOVABLE` due to language pre-training co-occurrences with motion words ('open', 'rotate'). Selective routing successfully cures this bias by resolving static infrastructure cheaply via geometry and domain priors.
2. **Computational Gating vs Classification Improvement**: On static test instances, estimator-aware routing (R3) achieves classification parity with non-estimator-aware routing (R2). The primary empirical contribution of estimator awareness is compute reduction: preventing expensive VLM inference on visual clutter that does not constrain the state estimator.
3. **Downstream Landmark Stability**: Heuristic geometric gating (R1) triggers false invalidations on peripheral landmarks subjected to visual odometry drift.

## 7.2 Explicit Limitations & Unresolved Bottlenecks
1. **Front-End Tracker Fragmentation**: 82.44% of upstream ByteTrack tracks last $\le 4$ frames. Intra-session consolidation agglomerates tracks locally, but recovering multi-session identity remains bottlenecked at 7/30 instances. Resolving this requires dense 3D instance point cloud tracking upstream of `vlm/`.
2. **2D Bounding Box Projection Jitter**: Derive centroids from 2D bounding box midpoints introduces depth projection jitter, contributing to a 22.28% relative displacement error.
3. **Semantic Coverage Constraint**: While matched-support Macro-F1 is 100%, observation coverage is 55.57% (3,535 unmapped out-of-vocabulary detector records).
4. **Trajectory ATE/RPE Ground Truth**: The physical room recording lacks external 6-DoF optical motion capture ground truth.

## 7.3 Prioritized Implementation Roadmap
1. **Phase 1 (Q1)**: Ingest dense 3D point cloud segmentation (FastSAM / Segment Anything 3D) to eliminate 2D bounding box projection jitter.
2. **Phase 2 (Q2)**: Integrate real-time Kalman tracking for 3D oriented bounding boxes to reduce front-end track fragmentation below 25%.
3. **Phase 3 (Q3)**: Collect synchronized Vicon/OptiTrack 6-DoF optical motion capture sequences to evaluate closed-loop trajectory ATE/RPE with live VLM feedback.
