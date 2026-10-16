### Table 13 Limitations And Boundaries: Identified System Limitations, Scientific Boundaries, and Mitigations

| Subsystem | Observed Limitation | Scientific Impact | Applied Mitigation | Future Work |
| :--- | :---: | :---: | :---: | :---: |
| Front-End Tracking | 82.44% tracks <= 4 frames | Cross-session instance recall bounded at 23.08% | Intra-session cluster consolidation | Dense 3D point cloud segmentation (FastSAM) |
| Relocation Geometry | 22.28% mean relative displacement error | Sub-centimeter precision unachievable with 2D box projection | Depth-weighted MAD outlier rejection | Direct 3D bounding box Kalman filter |
| Semantic Coverage | 55.57% observation coverage (3,535 unmapped) | Headline F1 (100%) applies only to matched support | Explicit reporting of coverage alongside F1 | Open-vocabulary open-world detector expansion |
| Downstream Trajectory | No 6-DoF optical motion capture GT on custom sequence | Continuous trajectory ATE/RPE cannot be reported | Fail-closed reporting (ATE/RPE marked UNEVALUATED) | Vicon/OptiTrack room recording |
| Visual Occlusion | Foreground obstacles in bounding box crops | Zero-shot VLM prompt noise | Multi-view crop aggregation | Instance mask alpha cropping |
