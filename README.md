# Dynamic 3D Scene Graph: Persistent Identity, Contextual Affordance & Multi-View VLM Reasoning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Testing: Pytest](https://img.shields.io/badge/tests-86%20passed-brightgreen.svg)]()

A robust, uncertainty-aware framework for indoor robot perception that bridges 3D metric-semantic SLAM and vision-language model (VLM) reasoning. It maintains **persistent physical identity across independent SLAM sessions**, detects physical object relocation, grounds contextual object affordance (immovable fixtures vs. movable furniture vs. mobile agents), and selectively escalates complex scenes to multi-view VLMs without ground-truth leakage.

---

## Key Capabilities

1. **Persistent Cross-Session Identity Association:**
   - Global bipartite matching with spatial distance, extent, and semantic gates.
   - Symmetric ambiguity rejection that flags competing nearby candidates to prevent false merges.
   - Evaluated on an explicit, session-qualified annotation benchmark covering 52 physical instances and 113 unique session-track mappings.
2. **Robust Multi-Session SE(2) Registration:**
   - RANSAC-estimated floor-plane rigid alignment across independent SLAM sequences.
   - Empirical residual dispersion and 5-fold held-out cross-validation reporting.
   - Strictly separates aligned reference geometry from tentative, unaligned entities.
3. **Contextual Scene-Graph Fusion:**
   - Mathematically exact incremental running mean for 3D centroids and bounding extents.
   - Disentangles instantaneous motion (`STATIC`, `DISPLACED`, `DYNAMIC`), physical affordance (`IMMOVABLE`, `MOVABLE`, `MOBILE`), and landmark validity.
   - Rejects unaligned geometry while preserving physical identity in a `TENTATIVE` lifecycle state.
4. **Estimator-Aware Selective Multi-View Routing:**
   - Deterministic policy routing based on SLAM tracking uncertainty and visual support evidence.
   - Reduces expensive multi-view VLM query budgets by resolving canonical static objects cheaply via validated SLAM priors.
   - Tested for complete invariance against ground-truth label mutations.

---

## Repository Structure

```
.
├── scene_graph/                 # Core ontology, persistent entity, and graph definitions
│   ├── ontology/                # Entity types, lifecycle states, and attributes
│   └── scene_graph.py           # Core graph node and edge storage
├── vlm/
│   ├── benchmarks/              # Canonical evaluation benchmarks and manifests
│   │   ├── cross_session_identity_annotations.json  # 113 session-qualified tracks, 52 physical instances
│   │   ├── movability_ground_truth.json             # Canonical physical instances and verified attributes
│   │   └── router_evaluation_manifest.json          # 15 development evaluation instances
│   ├── src/                     # Core perception and reasoning modules
│   │   ├── cross_session_association.py       # Cross-session candidate matching & 3 baselines
│   │   ├── stage5_adapter.py                  # SLAM evidence adapter without ground-truth leakage
│   │   ├── router_policy.py                   # Pure deterministic routing policy
│   │   ├── selective_router.py                # VLM query dispatcher and cache manager
│   │   └── contextual_scene_graph_fusion.py   # Continuous scene-graph fusion & running mean
│   ├── scripts/                 # Reproducible evaluation and audit scripts
│   │   ├── evaluate_multiview_affordance.py   # Stage 7 multi-view affordance evaluation
│   │   └── audit_semantic_associations.py     # Cross-session association & registration audit
│   ├── outputs/                 # Machine-readable evaluation results and markdown audits
│   │   ├── multiview_affordance_results.json
│   │   ├── multiview_affordance_evaluation.md
│   │   ├── stage5_registration_audit.json
│   │   └── semantic_association_audit.md
│   └── tests/                   # Complete pytest suite (86+ unit & integration tests)
├── pyproject.toml               # Package configuration and dependencies
└── pytest.ini                   # Pytest test discovery settings
```

---

## Installation & Setup

### Prerequisites
- Python >= 3.10
- Virtual environment recommended (`venv` or `conda`)

### 1. Clone & Setup Environment
```bash
git clone https://github.com/YashBhardwaj21/dynamic-3d-scene-graph.git
cd dynamic-3d-scene-graph

python -m venv myenv
# Windows:
.\myenv\Scripts\activate
# Linux/macOS:
source myenv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"
```

---

## Running Tests

Execute the comprehensive test suite verifying registration, association, router policies, and scene-graph fusion:

```bash
pytest vlm/tests/ -v
```

All 86 tests run locally without network access or live VLM servers required (offline mode evaluates cached or deterministic reasoning paths).

---

## Reproducing Evaluations

### 1. Cross-Session Identity Association & Registration Audit
Evaluates the three cross-session matching baselines (Nearest Centroid, Hungarian Gated, and Proposed Ambiguity-Aware) and checks robust SE(2) registration residuals:

```bash
python vlm/scripts/audit_semantic_associations.py
```
Output report: `vlm/outputs/semantic_association_audit.md`.

### 2. Multi-View Affordance & Selective Router Evaluation
Evaluates multi-view affordance reasoning across experimental conditions A through F using genuine inference records:

```bash
python vlm/scripts/evaluate_multiview_affordance.py
```
Output artifacts:
- `vlm/outputs/multiview_affordance_results.json`
- `vlm/outputs/multiview_affordance_evaluation.md`

---

## Scientific Methodology & Benchmark Contracts

- **Zero Ground-Truth Leakage:** Policy routing decisions are strictly derived from runtime SLAM observations, observation counts, and RANSAC registration. Target labels (`movability_gt`, `current_motion_gt`, `displacement_gt`) are used only after predictions are finalized to compute reference metrics.
- **Explicit Denominators:** All precision, recall, and F1 metrics report both the numerator and denominator. Evaluated annotated matches, false merges, and unannotated background candidates are explicitly distinguished.
- **Relocation vs. Identity:** Object relocation (spatial displacement in common reference frame) is evaluated as a physical state attribute separate from topological instance identity.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
