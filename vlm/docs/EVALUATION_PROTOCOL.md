# Multimodal Perception & Router Evaluation Protocol

## 1. Executive Summary & Purpose

This protocol specifies the standards for evaluating the Vision-Language Model (VLM) subsystem, the Stage 5 evidence adapter, and the selective evidence routing policies in `dynamic-3d-scene-graph`.

The central research question being tested is:
> **Does estimator-aware selective VLM reasoning improve the reliability of SLAM observations in changing indoor environments compared with strong geometry, temporal, semantic, and non-estimator-aware routing baselines, while reducing unnecessary VLM inference?**

To answer this question rigorously, the evaluation framework enforces strict experimental isolation, prevents ground-truth leakage, distinguishes within-benchmark selection variance from sample size, and mandates separate reporting for matched image-eligible samples versus end-to-end system deployments.

---

## 2. Dual-View Evaluation Standard

To eliminate misleading comparisons where missing-input cases are silently omitted or where different methods are evaluated on different denominators, all benchmark evaluations must report two distinct views:

### View A: Matched Image-Eligible Comparison
- **Population**: Restricted strictly to physical instances with valid, verified visual assets (harvested multi-view crops and scenes).
- **Objective**: Evaluates multimodal visual reasoning and selective query allocation under identical visual context.
- **Rules**:
  - All compared methods (All-VLM, Selective Router, Random Baseline, Oracle) must evaluate the **exact same physical instances**.
  - Query budgets must be explicitly matched (e.g., Random Baseline evaluates at budget $k = \text{queries}_{\text{selective}}$).
  - Metrics: Balanced Accuracy, Macro-F1, per-class Recall, Query Rate, End-to-End Latency, Live Inference Latency.

### View B: End-to-End System Comparison
- **Population**: The complete population under test (e.g., all 15 instances in the balanced manifest, or all 52 instances in the full benchmark), including instances with missing images, occlusions, or sensor failures.
- **Objective**: Evaluates the deployability and reliability of the complete perception system.
- **Rules**:
  - Missing visual inputs, corrupted depth, or tracking dropouts must **not** be discarded from the denominator.
  - Instances where required visual context is missing must trigger safe deferral (`DEFER` / `UNKNOWN`) or cheap resolution (`RESOLVE_CHEAP`) based on prior/motion evidence.
  - Abstentions remain strictly in the metric denominator.
  - Metrics: Full-population Balanced Accuracy, Coverage Rate, Abstention Rate, False Cheap Resolution Rate, Invalid Evidence Escalation Rate.

```
+-------------------------------------------------------------------------------+
|                             FULL PHYSICAL OBJECT POPULATION                  |
|                        (N=15 Manifest / N=52 Complete GT Census)              |
+---------------------------------------+---------------------------------------+
|        VIEW A: IMAGE-ELIGIBLE         |          NON-IMAGE INSTANCES          |
|  (Instances with verified crops)      |  (Missing cameras, unobserved crops)   |
|                                       |                                       |
|  • All-VLM Direct Reasoning (R4/M3)   |  • Prior-based Resolution (R0)        |
|  • Selective Routing (R2 / R3)        |  • Pure Motion Gate (R1)              |
|  • Matched Budget Random Selection    |  • Safe Deferral / Abstention         |
|  • Oracle Diagnostic Upper Bound      |                                       |
+---------------------------------------+---------------------------------------+
|                               VIEW B: END-TO-END SYSTEM                      |
|                (Evaluates complete system across ALL instances)                |
+-------------------------------------------------------------------------------+
```

---

## 3. Disentangled Routing Policy Matrix (R0–R4)

The evaluation compares five explicitly decoupled policies to isolate the source of any performance gains:

| Policy Identifier | Policy Name | Decision Basis | VLM Query Behavior | Estimator-Aware? |
| :--- | :--- | :--- | :--- | :---: |
| **R0** | **Semantic Class Prior Only** | Static dictionary lookup mapping `semantic_class -> movability` | Zero VLM queries ($0\%$ query rate) | No |
| **R1** | **Geometry & Motion Only** | Upstream SLAM motion gates (`STATIC`, `DISPLACED`, `DYNAMIC`) without vision | Zero VLM queries ($0\%$ query rate). Displaced objects yield `UNKNOWN`. | Yes (pure SLAM) |
| **R2** | **Non-Estimator-Aware Selective Router** | Affordance ambiguity and displacement gates, without SLAM landmark influence | Queries VLM whenever affordance is ambiguous or displacement occurs | No |
| **R3** | **Estimator-Aware Selective Router** | Affordance ambiguity conditioned on non-zero estimator relevance ($I_{\text{estimator}} > 0$) and query utility | Queries VLM only when object state directly impacts SLAM trajectory optimization | **Yes** |
| **R4** | **All-Eligible VLM Reference** | Direct zero-shot VLM prompting on all instances with visual context | Dispatches VLM on 100% of image-eligible instances | No |

---

## 4. Baseline Rigor and Statistical Guidelines

### 4.1 Monte-Carlo Random Query Baseline
- The random query baseline evaluates query allocation at matched budget $k = \text{queries}_{\text{selective}}$ across **1,000 deterministic pseudo-random seeds**.
- **Crucial Reporting Rule**: The 1,000 iterations represent **within-benchmark query-selection variability** (Monte-Carlo seed variance over query combinations on the same benchmark), **not** 1,000 independent datasets.
- Reports must state: Mean Balanced Accuracy, Standard Deviation, Median, 5th Percentile ($p_{05}$), and 95th Percentile ($p_{95}$).

### 4.2 Oracle Query Diagnostic Baseline
- The oracle baseline is a **non-deployable theoretical diagnostic upper bound** that uses ground truth to identify which instances would benefit from VLM escalation.
- **Zero Leakage Rule**: Ground truth must never be accessible to R0, R1, R2, R3, or R4 at decision time. Unit tests (`test_stage5_adapter_zero_leakage`) enforce that evidence dictionaries contain no ground truth fields.

### 4.3 Semantic Prior (M0) and Counterexample Testing
- Achieving a high score on a small benchmark where objects happen to match canonical prototypes (e.g., standard chairs and desks) is a sanity check, not proof of generalization.
- The evaluation must test R0 on the **Counterexample & Out-of-Distribution Suite** (`counterexample_suite`), which includes:
  - Caster-wheeled rolling whiteboard vs wall-bolted whiteboard.
  - VESA-mounted monitor on wall arm vs freestanding desk monitor.
  - Pedestal floor fan vs hardwired ceiling fan.
  - Carried backpack vs static floor backpack.
  - Out-of-vocabulary semantic categories (e.g. fire extinguisher, coat rack).

---

## 5. Execution Modes & Latency Accounting

The evaluation infrastructure distinguishes two operating modes to ensure timing transparency:

### A. Replay Mode (`mode="replay"`)
- Evaluates frozen, immutable prediction records stored in versioned JSON cache files (`baseline_checkpoint.json` and `router_vlm_cache.json`).
- Prevents redundant, non-deterministic re-querying of expensive vision-language models during test suite runs.
- **Reporting Rule**: Latencies loaded from cache must be labeled as `cached_latency_ms` and must **never** be presented as live system measurements.

### B. Live Inference Mode (`mode="live"`)
- Issues real HTTP POST requests to the local VLM server (`http://127.0.0.1:8089/v1/chat/completions`).
- Measures true end-to-end wall-clock time using `time.perf_counter()`, logs prompt token count, completion token count, and total latency.

### Latency Metric Decomposition
For every evaluation condition, latency must be decomposed into:
1. **Decision Latency ($t_{\text{decision}}$)**: Time taken to evaluate evidence validity gates and routing rules (typically $0.05\text{--}0.20\text{ ms}$).
2. **Inference Latency ($t_{\text{inference}}$)**: Time spent in VLM forward-pass execution (typically $15\text{--}25\text{ s}$ per query on local CPU).
3. **End-to-End Latency ($t_{\text{e2e}}$)**: Total instance latency:
   $$t_{\text{e2e}} = t_{\text{decision}} + \mathbb{I}(\text{query}) \cdot t_{\text{inference}}$$
   Reported across both the full population (including unqueried $0\text{ ms}$ cases) and queried instances only.

---

## 6. Mathematical Formulations of Primary Metrics

### Balanced Accuracy
$$\text{Balanced Accuracy} = \frac{1}{|C|} \sum_{c \in C} \text{Recall}_c = \frac{1}{|C|} \sum_{c \in C} \frac{TP_c}{TP_c + FN_c}$$
Where $C = \{\text{IMMOVABLE}, \text{MOVABLE}, \text{MOBILE}\}$. Any prediction of `UNKNOWN` or abstention counts as a False Negative ($FN_c$) for the true class, correctly penalizing balanced accuracy.

### Macro-F1
$$\text{Macro-F1} = \frac{1}{|C|} \sum_{c \in C} F1_c, \quad F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

### Query Efficiency Metrics
- **VLM Query Rate**: $\frac{N_{\text{queried}}}{N_{\text{total}}} \times 100\%$
- **Queries Avoided**: $N_{\text{total}} - N_{\text{queried}}$
- **Abstention Rate**: $\frac{N_{\text{abstained}}}{N_{\text{total}}} \times 100\%$
- **Query Utility**: $U = \Delta \mathcal{L}_{\text{SLAM}} - \lambda_{\text{cost}} \cdot C_{\text{VLM}}$
