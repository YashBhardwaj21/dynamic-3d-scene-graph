# Chapter 2: Background and Literature Review

## 2.1 Spatial SLAM in Dynamic & Changing Environments
Metric Simultaneous Localization and Mapping (SLAM) has achieved high precision in static environments (Mur-Artal et al., ORB-SLAM3; Labbe & Michaud, RTAB-Map). However, persistent indoor operation over days or weeks breaks the static world assumption. Early dynamic SLAM formulations (e.g., DynaSLAM, Mask-SLAM) utilized 2D instance segmentation to mask moving objects (typically pedestrians and vehicles). While effective for active dynamics, masking removes useful visual constraints provided by stationary objects and cannot reason about objects relocated between passes.

## 2.2 3D Semantic Scene Graphs
Hierarchical 3D Scene Graphs (Armeni et al., 2019; Rosinol et al., Kimera, 2020; Hughes et al., Hydra, 2022) organize spatial environments into layered topological abstractions (metric meshes, object nodes, room spaces, and buildings). Hydra and Kimera maintain real-time metric-semantic graphs, but treat object nodes as static geometric clusters. When an object shifts, these systems either instantiate duplicate disconnected nodes or smooth the shifted positions into corrupted spatial centroids.

## 2.3 Multimodal Foundation Models for Spatial Perception
Recent systems such as ConceptGraphs (Gu et al., 2023) and SayPlan (Rana et al., 2023) harness 2D Vision-Language Models (VLMs) and CLIP embeddings to build open-vocabulary 3D scene graphs. ConceptGraphs constructs open-vocabulary object representations by associating segmented masks with CLIP embeddings. However, ConceptGraphs operates in batch offline mode and averages multi-session centroids indiscriminately, generating phantom centroids for relocated objects.

## 2.4 Selective & Cost-Aware Foundation Model Dispatch
Executing foundation model forward passes (e.g., Qwen2.5-VL-3B, GPT-4o) on every camera frame is computationally intractable on mobile robotics platforms, where inference latencies range from 2 to 25 seconds per query. Selective inference methods (e.g., FrugalGPT, Cascade Routing) prioritize cheap heuristics and escalate only high-uncertainty instances. In spatial SLAM, however, semantic uncertainty alone is insufficient: an ambiguous object in the far background that provides zero constraints to the pose graph does not justify expensive VLM reasoning. This research addresses this gap by formalizing **estimator-aware query utility**.

## 2.5 Summary of Methodological Distinctions
Table 2.1 in the reporting package provides a systematic comparison between the proposed method and leading spatial semantic systems (Kimera, Hydra, ConceptGraphs, SayPlan). The primary distinction is the explicit coupling of state estimator landmark relevance with multimodal affordance verification.
