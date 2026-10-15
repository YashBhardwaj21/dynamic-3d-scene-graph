# Ground-Truth Benchmark Protocol & Dataset Manifest Architecture

## 1. Executive Summary & Objective

This document defines the authoritative ground-truth benchmark protocol for evaluating multimodal perception, object movability, dynamic scene graph maintenance, and estimator-aware selective VLM routing in the `dynamic-3d-scene-graph` repository.

Without verified physical ground truth, model outputs remain uncalibrated predictions. Following the methodological framework of **Rüdt et al.** (*Towards Dynamic 3D Scene Graphs: Finding and Tracking Objects in the Wild*, arXiv:2404.09904), this benchmark:
1. Implements a three-question disentangled ontology separating intrinsic physical affordances (`movability_gt`) from instantaneous motion (`current_motion_gt`) and downstream metric landmark validity (`slam_invalidated`).
2. Establishes independent ground truth via dual human annotation, consensus adjudication (Cohen's $\kappa = 0.8587$), and physical steel-tape measurement ($\sigma = 0.02\text{ m}$) corroborated by RANSAC SE(2) rigid anchor registration ($\sigma = 0.1198\text{ m}$).
3. Catalogs all physical instances in an authoritative machine-readable manifest (`vlm/benchmarks/canonical_benchmark_manifest.json`) with cryptographic SHA-256 hashes of all visual crop assets.
4. Formulates explicit, non-overlapping dataset populations to prevent conflating image-eligible sample subsets with end-to-end system populations.

---

## 2. Disentangled Three-Question Ontology

To prevent the common conflation of physical affordance with momentary observation, the benchmark evaluates three fundamentally distinct quantities for every object instance:

| Question | Conceptual Target | Benchmark Field | Permissible Values |
| :--- | :--- | :--- | :--- |
| **Q1: Can this object be relocated?** | Contextual physical affordance / movability prior | `movability_gt` | `IMMOVABLE`, `MOVABLE`, `MOBILE`, `UNKNOWN` |
| **Q2: Did it physically move during the observation interval?** | Observed spatial displacement | `current_motion_gt` | `STATIC`, `DISPLACED`, `DYNAMIC`, `UNCERTAIN` |
| **Q3: Did its displacement make prior SLAM landmarks invalid?** | Epistemic landmark reliability for geometric SLAM | `slam_invalidated` | `true`, `false` |

### A. Contextual Movability (`movability_gt`)
The three canonical categories established by Rüdt et al. are rigorously maintained:
- **`IMMOVABLE`**: Fixed architectural infrastructure or structural elements that cannot ordinarily be relocated without tools, demolition, or building alterations.
  - *Examples*: Architectural walls, interior passage doors framed in jambs, ceiling fans wired and anchored to concrete slabs, structural pillars.
- **`MOVABLE`**: Freestanding physical objects that can ordinarily be relocated, pushed, or carried by external human or robotic agents without structural disassembly.
  - *Examples*: Caster-wheeled office chairs, workstation desks, storage boxes, laptops, backpacks, computer monitors, pedestal fans.
  - *Governing Principle*: A stationary chair is still `MOVABLE`. Movability is an inherent physical affordance and contextual capability, not a function of whether motion occurred during a specific observation window.
- **`MOBILE`**: Contextual category involving intrinsic locomotion capability or active mobile carrier dynamics.
  - *Examples*: Autonomous robotic platforms (e.g. 4-motor RC car), human occupants/pedestrians, quadcopter drones, or objects actively carried by an agent.
  - *Governing Principle*: `MOBILE` implies intrinsic propulsion or carrier dynamics; it is strictly not a synonym for "displaced in this frame".
- **`UNKNOWN` / Abstention**: Explicit category reserved for perceptual uncertainty when visual evidence is insufficient or corrupted.

### B. Observed Motion (`current_motion_gt`)
- **`STATIC`**: The physical object maintained a constant world-relative 3D position within combined measurement uncertainty ($\Delta \mathbf{p} \le 3\sigma_{\text{total}} = 0.364\text{ m}$) throughout the recorded interval.
- **`DISPLACED`**: The physical object was relocated between sessions ($\Delta \mathbf{p} > 3\sigma_{\text{total}}$).
- **`DYNAMIC`**: The entity underwent active, continuous motion within a single recording session (e.g., walking human, moving robot).

### C. SLAM Landmark Invalidation (`slam_invalidated`)
- **`true`**: The physical object changed position between observation epochs such that past geometric landmark observations are inconsistent with current reality and must be culled or downweighted in SLAM optimization.
- **`false`**: The object remained static; prior geometric observations remain valid metric constraints for map optimization and loop closure.

---

## 3. Physical Measurement & Uncertainty Quantification

Controlled physical relocations were executed across two recording sessions (`semantic_mapping_my_desk_recording_01` and `semantic_mapping_my_desk_recording_02`) under calibrated metric verification:

```
[Session 1: Baseline Traversal]
   ├── Sensor platform maps stationary room
   └── 3D centroids logged for all physical objects

[Controlled Physical Intervention]
   ├── Calibrated steel tape measure records metric displacement: sigma_tape = 0.020 m
   └── Stationary reference anchors maintained untouched

[Session 2: Re-observation Traversal]
   ├── Sensor platform re-traverses environment
   └── Displaced objects, stationary anchors, and novel entities observed

[Rigid SE(2) Registration Verification]
   ├── Distance-preserving RANSAC SE(2) registration solves inter-session alignment
   ├── Inlier registration residual: sigma_reg = 0.1198 m
   └── Combined measurement uncertainty: sigma_total = sqrt(0.02^2 + 0.1198^2) = 0.1215 m
```

Statistically significant motion events require:
$$\Delta \mathbf{p}_{\text{world}} \ge 3 \sigma_{\text{total}} \approx 0.364\text{ m}$$

---

## 4. Benchmark Populations and Split Specifications

The repository contains multiple evaluation populations that serve distinct experimental objectives and **must not be treated as interchangeable**:

| Population Identifier | Size | Composition | Visual Assets | Purpose |
| :--- | :---: | :--- | :--- | :--- |
| **`pop_52_ground_truth`** | 52 | 4 IMMOVABLE, 45 MOVABLE, 3 MOBILE | 39 with images, 13 without images | Comprehensive physical instance census in `movability_ground_truth.json`. Evaluates full end-to-end system capability (View B). |
| **`pop_39_image_eligible`** | 39 | 2 IMMOVABLE, 37 MOVABLE, 0 MOBILE | 100% have harvested multi-view crops | Evaluates vision-based affordance reasoning under matched visual inputs (View A). Genuine Qwen2.5-VL-3B inferences recorded. |
| **`pop_15_balanced_router`** | 15 | 3 IMMOVABLE, 9 MOVABLE, 3 MOBILE | 11 with images, 4 without images | Balanced Stage 6 manifest (`router_evaluation_manifest.json`) designed for router policy comparison (R0–R4). |
| **`pop_28_held_out_furniture`**| 28 | 0 IMMOVABLE, 28 MOVABLE, 0 MOBILE | 100% have harvested crops | Strictly held-out image instances outside the 15-instance manifest ($39 - 11 = 28$). Note: 100% movable furniture. |
| **`counterexample_suite`** | 10 | Context-dependent edge cases | Specification & test fixtures | Tests where static semantic class priors fail (e.g. rolling whiteboard, VESA-mounted monitor, carried backpack, unseen classes). |

### Split Leakage Prevention
- Every physical object has an immutable `entity_id` (e.g. `inst_chair_002`).
- Track IDs (`chair_002`, `chair_001`) from different sessions are grouped under the single physical `entity_id`.
- The 28 held-out instances share zero physical objects with the 15-instance router development manifest.

---

## 5. Dual-Annotator Verification & Adjudication

To eliminate annotator subjectivity:
1. Two domain experts independently annotated all physical instances:
   - **Annotator A**: Robotics & State Estimation Specialist (`annotator_a_robotics.json`).
   - **Annotator B**: Computer Vision & Multimodal Perception Specialist (`annotator_b_vision.json`).
2. Inter-annotator agreement metrics:
   - **Raw Agreement**: $96.15\%$ (50 / 52 instances in unanimous agreement).
   - **Cohen's Kappa ($\kappa$)**: $0.8587$ (indicating near-perfect agreement beyond chance).
3. Adjudication Protocol:
   - The 2 disputed instances (`inst_ceiling_fan_001` and `inst_pedestal_fan_001`) were adjudicated through joint physical inspection of electrical conduit wiring and mechanical mounting:
     - Ceiling fan: Confirmed `IMMOVABLE` (hardwired to concrete ceiling slab).
     - Pedestal fan: Confirmed `MOVABLE` (freestanding floor appliance with plug-in cord).
   - Consensus decisions recorded in `vlm/benchmarks/annotation_adjudication.json`.

---

## 6. Cryptographic Asset Integrity

All harvested visual crops and scene context images are indexed in `vlm/benchmarks/canonical_benchmark_manifest.json` with SHA-256 hashes. Any modification or corruption of crop images triggers automated cache invalidation, guaranteeing 100% reproducible and tamper-free evaluation.
