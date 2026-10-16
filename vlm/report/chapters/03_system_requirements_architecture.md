# Chapter 3: System Requirements and Architecture

## 3.1 Functional Requirements
- **FR-01 (Pre-Router Evidence Extraction)**: Ingest raw detector streams and extract 3D geometric point-to-box consistency, temporal persistence, and Shannon entropy without triggering foundation models. Target: > 100 rec/s (Measured: 1,061.9 rec/s).
- **FR-02 (Multi-Session Registration)**: Compute rigid SE(2) consensus alignment between distinct mapping trajectories. Target: consensus inlier RMSE < 0.20 m (Measured: 0.1198 m).
- **FR-03 (Selective Routing Decoupling)**: Provide reproducible execution for five distinct policies (R0, R1, R2, R3, R4) over identical observation streams.
- **FR-04 (Decoupled Scene Graph Maintenance)**: Maintain decoupled historical pose records for displaced entities to prevent phantom centroid drift.

## 3.2 Non-Functional Requirements & Engineering Guardrails
- **NFR-01 (Fail-Closed Integrity)**: Trajectory ATE/RPE error metrics must report as UNEVALUATED (`null`) when external 6-DoF optical motion capture ground truth is unavailable.
- **NFR-02 (Deterministic Execution)**: All benchmark evaluations must execute with fixed random seeds and verifiable SHA-256 provenance hashes.
- **NFR-03 (Memory & Compute Bounds)**: Peak resident memory must remain below 16 GB RAM during full 7,956-record pipeline execution.

## 3.3 System Architecture
Figure 3.1 depicts the end-to-end architecture of the Dynamic 3D Scene Graph pipeline:
1. **Sensor Ingestion & Upstream Tracker**: Ingests synchronized RGB-D frames, camera poses from RTAB-Map, and 2D YOLO bounding box detections.
2. **Stage 5 Inexpensive Pre-Router Pipeline**:
   - `GeometricEvidenceExtractor`: Accumulates depth distributions, checks bounding box inlier ratios, and computes depth variance.
   - `TemporalEvidenceExtractor`: Tracks object lifespan, persistence scores, and occlusion recovery.
   - `SemanticEvidenceExtractor`: Computes multi-frame class vote purity and Shannon entropy.
3. **Stage 6 Estimator-Aware Selective Router**: Evaluates query utility $U(q)$ against inference cost $\lambda_{\text{cost}} C_{\text{VLM}}$.
4. **Cross-Session Association & Scene Graph Fusion Engine**: Agglomerates intra-session clusters and maintains decoupled landmark nodes.
