# Vision-Language Model (VLM) Subsystem for Dynamic 3D Scene Graph

## 1. Overview & Governing Research Objective

This directory contains the Vision-Language Model (VLM) subsystem, multimodal evidence fusion adapter, decoupled evidence routing policies, and evaluation framework for **Dynamic 3D Scene Graph**.

### Governing Research Question
> **Does estimator-aware selective VLM reasoning improve the reliability of SLAM observations in changing indoor environments compared with strong geometry, temporal, semantic, and non-estimator-aware routing baselines, while reducing unnecessary VLM inference?**

Following the foundational methodology of **Rüdt et al.** (*Towards Dynamic 3D Scene Graphs: Finding and Tracking Objects in the Wild*, arXiv:2404.09904), this subsystem disentangles three fundamentally distinct questions:
1. **Contextual Movability (`movability_gt`)**: Inherent physical affordance (`IMMOVABLE`, `MOVABLE`, `MOBILE`).
2. **Current Motion (`current_motion_gt`)**: Observed spatial displacement (`STATIC`, `DISPLACED`, `DYNAMIC`).
3. **Downstream Landmark Reliability (`slam_invalidated`)**: Whether prior geometric observations remain valid metric constraints or must be culled to protect SLAM trajectory optimization.

---

## 2. Directory Architecture

```
vlm/
├── benchmarks/                     # Authoritative ground-truth datasets & manifests
│   ├── canonical_benchmark_manifest.json  # Comprehensive catalog of 53 entities with SHA-256 hashes
│   ├── router_evaluation_manifest.json    # Balanced 15-instance router development manifest
│   ├── movability_ground_truth.json       # 52 verified physical object census
│   ├── controlled_scene_events.json       # Metric tape-measured relocation ground truth
│   └── cross_session_identity_annotations.json # Multi-session track associations
├── docs/                           # Authoritative engineering and research documentation
│   ├── ENGINEERING_AUDIT.md        # Comprehensive system audit and starting state record
│   ├── ISSUE_REGISTER.md           # Defect register tracking Issues 01-10 through resolution
│   ├── DATA_AND_EVALUATION_INVENTORY.md # Catalog of source datasets, tracks, and populations
│   ├── PIPELINE_AND_DATA_FLOW.md   # Architectural data flow and evidence pipeline
│   ├── PROMPT_SPECIFICATION.md     # Neutral ontology schema prompt contract (v2)
│   ├── GROUND_TRUTH_PROTOCOL.md    # Multi-session ground-truth and uncertainty protocol
│   ├── EVALUATION_PROTOCOL.md      # Dual-view standards, R0-R4 policies, and metric accounting
│   ├── METRIC_DICTIONARY.md        # Formulations, denominators, and caveats for all metrics
│   ├── EXPERIMENT_LOG.md           # Chronological experimental run records
│   ├── RESEARCH_RESULTS.md         # Direct answers to research questions with verified evidence
│   └── REPRODUCIBILITY.md          # Exact reproduction commands and environment specs
├── outputs/                        # Versioned evaluation deliverables and reports
│   ├── baseline_checkpoint.json    # Genuine Qwen2.5-VL-3B inferences on 39 instances (M1/M2/M3)
│   ├── stage6/                     # Stage 6 router ablation outputs
│   │   ├── router_ablation_results.json
│   │   └── router_evaluation_report.md
│   ├── full_experiment_run/        # Complete 7,956-observation pipeline run
│   │   ├── scene_graph.json        # 195 persistent 3D nodes with decoupled pose histories
│   │   └── association_baselines_summary.json
│   ├── vlm_controlled_ablations.json # Phase 6 prompt, view-count, and input representation ablations
│   └── downstream_slam_effects.json  # Phase 8 landmark reliability and ATE/RPE trajectory metrics
├── scripts/                        # Evaluation, ablation, and orchestrator scripts
│   ├── run_full_experiment.py      # End-to-end multi-session pipeline orchestrator
│   ├── run_stage6_ablation.py      # Decoupled router ablation runner (R0-R4, Random, Oracle)
│   ├── run_vlm_controlled_ablations.py # Phase 6 prompt and view-count ablation runner
│   ├── evaluate_downstream_slam_effects.py # Phase 8 downstream SLAM and scene graph evaluator
│   └── generate_canonical_manifest.py  # Generates cryptographic benchmark manifest
├── src/                            # Core library implementations
│   ├── router_policy.py            # Decoupled routing policies (R0, R1, R2, R3, R4)
│   ├── selective_router.py         # Instance aggregator, VLM dispatcher, and PIL validator
│   ├── stage5_adapter.py           # Multi-session evidence extraction and uncertainty propagation
│   ├── contextual_scene_graph_fusion.py # Persistent scene graph nodes and landmark invalidation
│   └── cross_session_association.py     # SE(2) registered multi-session matching baselines
└── tests/                          # 113 unit and regression tests (100% passing)
```

---

## 3. Decoupled Routing Policies (R0–R4)

| Policy | Name | Description | VLM Query Rate |
| :---: | :--- | :--- | :---: |
| **R0** | Semantic Prior | Static dictionary lookup mapping `semantic_class -> movability` | `0.0%` |
| **R1** | Geometry & Motion Only | Inexpensive SLAM motion gates without vision | `0.0%` |
| **R2** | Non-Estimator-Aware | Escalates on affordance ambiguity regardless of estimator influence | `33.3%` |
| **R3** | Estimator-Aware | Escalates only when $I_{\text{estimator}} > 0$ and query utility $U > 0$ | `33.3%` |
| **R4** | All-Eligible VLM | Direct zero-shot VLM prompting on all image-eligible instances | `73.3%` |

---

## 4. Key Experimental Results

1. **Curing the Standalone VLM Mobility Bias**: Zero-shot VLM prompting (R4) incorrectly predicts `MOVABLE` for architectural fixtures (`door`, `ceiling fan`) due to strong linguistic associations with dynamic motion ('open/close', 'rotate/spin'). Selective routing (R3) resolves architectural structures cheaply via geometry/prior, achieving higher accuracy than All-VLM.
2. **Computational Query Reduction**: Estimator-aware routing avoids **54.5%–63.6%** of expensive VLM calls on image-eligible instances, speeding up mean latency from **26.33 s** to **13.64 s** (a **1.93x** speedup).
3. **Protecting Downstream SLAM Observations**: Stationary movable objects (e.g., 17 stationary chairs) are correctly retained as metric landmarks (100% retention rate, 0% false invalidation), while stale observations of confirmed relocated objects are invalidated to prevent map distortion.
4. **Semantic Prior Failure on Counterexamples**: Naive dictionary lookup (R0) drops from 85.2% on prototype furniture to **30.0%** on contextual counterexamples (rolling whiteboards, VESA-mounted monitors, carried backpacks, and out-of-vocabulary classes). Selective contextual reasoning achieves **90.0%** on these cases.

---

## 5. Quickstart & Reproduction Commands

To run the complete reproducible evaluation suite:

```bash
# 1. Run all unit and regression tests (113 tests)
py -3 -m pytest vlm/tests -q

# 2. Build the cryptographic benchmark manifest
py -3 vlm/scripts/generate_canonical_manifest.py

# 3. Run Stage 6 decoupled router policy ablations (R0-R4)
py -3 vlm/scripts/run_stage6_ablation.py

# 4. Run Phase 6 prompt, view-count, and input representation ablations
py -3 vlm/scripts/run_vlm_controlled_ablations.py

# 5. Run the complete multi-session pipeline orchestrator (7,956 records -> 195 scene graph nodes)
py -3 vlm/scripts/run_full_experiment.py

# 6. Evaluate downstream SLAM landmark reliability and trajectory error
py -3 vlm/scripts/evaluate_downstream_slam_effects.py
```
