# VLM Data Package (`vlm/data/`)

This directory contains the self-contained data artifacts, multi-view instance observations, trajectory logs, and semantic mapping records required to run, test, and reproduce the Vision-Language Model (VLM) research branch (`research-vlm`).

## 1. Directory Structure

```
vlm/data/
├── vlm_data_manifest.json           # SHA-256 cryptographic manifest for all 751 files
├── raw/                             # Trajectory logs and sequence metadata
│   ├── my_desk_recording_01/        # Session 1: associations.txt, groundtruth.txt, intrinsics, etc.
│   └── my_desk_recording_02/        # Session 2: associations.txt, groundtruth.txt, intrinsics, etc.
├── semantic_mapping_desk_run_20261009_161552/   # Session 1 semantic map & entities
├── semantic_mapping_d455_rgbd_20261010_123443/  # Session 2 semantic map & entities
├── semantic_mapping_fused_desk_run/             # Fused multi-session reference map
├── semantic_mapping_my_desk_recording_01/       # Session 1 MapPoints.ply & observations
├── semantic_mapping_my_desk_recording_02/       # Session 2 MapPoints.ply & observations
├── multiview_samples/               # 39 physical object instances (crops, context, metadata)
│   ├── inst_backpack_001/
│   ├── inst_chair_001/ ... inst_chair_035/
│   ├── inst_desk_001/ ... inst_desk_031/
│   ├── inst_door_001/
│   ├── inst_laptop_001/
│   ├── inst_monitor_001/
│   ├── inst_pedestal_fan_001/
│   ├── inst_storage_box_001/ ... inst_storage_box_007/
│   └── inst_whiteboard_001/
└── observations/                    # Unified multi-session observation cache
    ├── object_observations.jsonl    # 7,956 object detection instances across sessions
    ├── object_tracks.jsonl          # 78 confirmed object tracks
    └── multiview_association_montage.jpg
```

## 2. File Provenance & Original Locations

| `vlm/data/` Subdirectory | Original Source Location | Purpose & Consumer |
| :--- | :--- | :--- |
| `raw/my_desk_recording_01/` | `data/raw/my_desk_recording_01/` | Session 1 camera poses, timestamp associations, and groundtruth trajectory. Used by `crop_harvester.py` and `cross_session_association.py`. |
| `raw/my_desk_recording_02/` | `data/raw/my_desk_recording_02/` | Session 2 camera poses, timestamp associations, and groundtruth trajectory. Used for multi-session alignment. |
| `semantic_mapping_desk_run_20261009_161552/` | `outputs/semantic_mapping_desk_run_20261009_161552/` | Initial desk recording semantic entities and spatial clusters. Used by `audit_and_extract_entities.py`. |
| `semantic_mapping_d455_rgbd_20261010_123443/` | `outputs/semantic_mapping_d455_rgbd_20261010_123443/` | Second recording semantic entities. Used by `audit_and_extract_entities.py`. |
| `semantic_mapping_fused_desk_run/` | `outputs/semantic_mapping_fused_desk_run/` | Unified multi-session ground truth registration and floor plane. Used by `generate_ground_truth_benchmark.py`. |
| `semantic_mapping_my_desk_recording_01/` | `outputs/semantic_mapping_my_desk_recording_01/` | Session 1 MapPoints (`trajectory/MapPoints.ply`), occupancy maps, and observations. Used by `stage5_pipeline.py` and `geometric_evidence.py`. |
| `semantic_mapping_my_desk_recording_02/` | `outputs/semantic_mapping_my_desk_recording_02/` | Session 2 MapPoints (`trajectory/MapPoints.ply`), occupancy maps, and observations. Used by `stage5_pipeline.py` and `geometric_evidence.py`. |
| `multiview_samples/` | `vlm/outputs/multiview_samples/` | 39 physical object instances with viewpoint crops (`crop_view_*.jpg`), full frame context (`scene_view_*.jpg`), and observation metadata. Used by `selective_router.py` and `test_router_metrics.py`. |
| `observations/` | `vlm/outputs/` | Pre-extracted 7,956 observations and tracks cache. Used by `stage5_pipeline.py` and `test_scene_graph_integration.py`. |

## 3. Large File Handling & GitHub Limits

- **Excluded Raw Video Streams**: The full uncompressed RGB and depth image streams for the two raw sequences comprise over 21,000 PNG files totaling **~5.95 GB** (`my_desk_recording_01`: 2.89 GB; `my_desk_recording_02`: 3.06 GB). Storing 6 GB of video frames directly in git exceeds standard GitHub repository size recommendations (< 2 GB) and causes clone bloat.
- **Included Lightweight Artifacts**: Instead, all camera intrinsics, timestamp synchronizations, trajectory logs, and the **exact extracted multi-view crops** for every physical entity (166.88 MB total) are packaged directly in `vlm/data/`. This enables complete offline execution of all VLM routing, ablation, benchmark scoring, and downstream SLAM tests without needing the 6 GB raw image streams.
- **Excluded Mesh Reconstructions**: High-density Poisson reconstruction meshes (`world_objects.ply`, ~31 MB) from mapping runs are omitted since the VLM pipeline consumes sparse ORB-SLAM3 landmark point clouds (`trajectory/MapPoints.ply`, ~400 KB) and bounding box observations.
