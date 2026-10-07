# Dynamic 3D Scene Graph — Forensic Audit & Refactor Inventory

## Overview
This document contains the complete file-by-file forensic audit of the `dynamic-3d-scene-graph` repository prior to the structural reorganization. It establishes the single authoritative real-time pipeline for the Intel RealSense D455 on Jetson Orin Nano with RTAB-Map SLAM and YOLOE perception, while safely archiving historical TUM, my_desk_sequence, and legacy Windows-WSL transport implementations.

---

## 1. Current Runtime Dependency Graph

```mermaid
graph TD
    subgraph Active Live Robotics Pipeline
        D455[D455 RealSense Camera via USB 3.x] -->|RGB-D 640x480 @ 30Hz + IMU @ 200Hz| ROS_TOPICS[/camera/camera/color/image_raw<br>/camera/camera/aligned_depth_to_color/image_raw<br>/camera/camera/imu]
        ROS_TOPICS --> RTABMAP[RTAB-Map Visual Odometry & SLAM Node]
        ROS_TOPICS --> SG_NODE[scene_graph_node: ROS Ingestion & Sync]
        RTABMAP -->|TF: world -> camera_color_optical_frame<br>Odometry & Loop Closures| SG_NODE
        
        SG_NODE -->|FramePacket: rgb, depth_m, world_T_camera, K| YOLOE[YOLOEDetector: Open-Vocab 2D Mask/BBox Segmentation]
        YOLOE -->|List[Observation]| CORE_PIPE[SceneGraphPipeline: pipeline_core.py]
        
        subgraph Canonical Core Algorithm
            CORE_PIPE --> GEOM[3D Geometry: Point Cloud Centroid, OBB, Noise Model, Surfaces]
            GEOM --> TRACKER[CausalTracker: Kalman 3D Tracking & ID Continuity]
            TRACKER --> OBJECT_STATE[ObjectStateMachine: ACTIVE, UNOBSERVED, OCCLUDED]
            OBJECT_STATE --> CANDIDATE_GEN[RelationCandidateGenerator]
            CANDIDATE_GEN --> REL_REGISTRY[RelationRegistry: Distance, Support, Directional, Containment, Occlusion]
            REL_REGISTRY --> REL_STATE[RelationStateMachine: Evidence Accumulation & Decay]
            REL_STATE --> TEMP_GRAPH[TemporalSceneGraph: Persistent Nodes, Edges, Provenance]
        end
        
        TEMP_GRAPH --> GRAPH_PUB[GraphPublisher: MarkerArray, Scene State JSON, PointClouds]
        GRAPH_PUB --> RVIZ[RViz2 3D Scene Graph Visualization]
        GRAPH_PUB --> VIEWER[live_2d_viewer: 2D Perception & Track Dashboard]
    end

    subgraph Historical / Replay Paths to Archive
        TUM_DATA[(TUM Freiburg RGB-D Dataset)] -.-> TUM_PLAYER[tum_player.py / tum_source.py]
        TUM_PLAYER -.-> TUM_LAUNCH[tum_scene_graph.launch.py]
        DESK_DATA[(my_desk_sequence)] -.-> BENCH_SCRIPTS[run_stage3_evaluation.sh]
        WIN_RS[Windows pyrealsense2] -.-> D455_SENDER[d455_sender.py: TCP Port 5000]
        D455_SENDER -.-> D455_RECV[d455_receiver.py / d455_bridge]
    end
```

---

## 2. File-by-File Inventory & Classification

| Current Path | Role | Runtime Critical | Dataset Specific | Duplicate/Legacy | Destination | Action | Rationale |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :--- |
| `configs/default.yaml` | System-wide base configuration | Yes | No | No | `configs/runtime/default.yaml` | MOVE | Core base configuration schema. |
| `configs/live_d455.yaml` | Live D455 + RTAB-Map config | Yes | No | No | `configs/runtime/live_d455.yaml` | MOVE | Authoritative live deployment config. |
| `configs/sensors/realsense_d455.yaml` | Sensor noise and baseline parameters | Yes | No | No | `configs/sensors/realsense_d455.yaml` | KEEP | Reusable sensor physical parameter config. |
| `configs/tum_fr1_desk.yaml` | TUM Freiburg 1 desk config | No | Yes (TUM) | Historical | `archive/tum/configs/tum_fr1_desk.yaml` | ARCHIVE | Dataset-specific replay experiment. |
| `configs/tum_validation.yaml` | TUM validation experiment config | No | Yes (TUM) | Historical | `archive/tum/configs/tum_validation.yaml` | ARCHIVE | Dataset-specific offline evaluation config. |
| `configs/my_desk_sequence.yaml` | Local desk recorded replay config | No | Yes (my_desk) | Historical | `archive/my_desk_sequence/config/my_desk_sequence.yaml` | ARCHIVE | Local recorded experiment, not live runtime. |
| `src/scene_graph/__init__.py` | Package root initialization | Yes | No | No | `src/scene_graph/__init__.py` | KEEP | Authoritative package entry. |
| `src/scene_graph/config.py` | Config dataclasses & YAML parser | Yes | No | No | `src/scene_graph/config.py` | KEEP | Authoritative type-safe configuration classes. |
| `src/scene_graph/data/__init__.py` | Data module exports | Yes | No | Legacy exports | `src/scene_graph/data/__init__.py` | REFACTOR | Remove `tum_loader` / `tum_source` exports. |
| `src/scene_graph/data/frame_packet.py` | Canonical frame representation | Yes | No | No | `src/scene_graph/data/frame_packet.py` | KEEP | Authoritative multi-modal frame contract. |
| `src/scene_graph/data/frame_source.py` | Abstract frame source interface | Yes | No | No | `src/scene_graph/data/frame_source.py` | KEEP | Base protocol for live and replay adapters. |
| `src/scene_graph/data/pose_estimate.py` | PoseEstimate & EstimatorState | Yes | No | No | `src/scene_graph/data/pose_estimate.py` | KEEP | Authoritative SE(3) pose & tracking state. |
| `src/scene_graph/data/sensor_frame.py` | Ingestion sensor frame contract | Yes | No | No | `src/scene_graph/data/sensor_frame.py` | KEEP | Stage 1 sensor contract with depth scale. |
| `src/scene_graph/data/synchronization.py` | Nearest-neighbor time association | Yes | No | No | `src/scene_graph/data/synchronization.py` | KEEP | Generic timestamp association algorithm. |
| `src/scene_graph/data/timestamp.py` | Monotonic & domain timestamp types | Yes | No | No | `src/scene_graph/data/timestamp.py` | KEEP | Authoritative clock domain representation. |
| `src/scene_graph/data/tum_loader.py` | Parser for TUM format text files | No | Yes (TUM) | Dataset adapter | `archive/tum/loaders/tum_loader.py` | ARCHIVE | Dataset-specific loader; not active core. |
| `src/scene_graph/data/tum_source.py` | Generator of FramePackets from TUM | No | Yes (TUM) | Dataset adapter | `archive/tum/replay/tum_source.py` | ARCHIVE | Replay adapter for TUM sequences. |
| `src/scene_graph/estimation/__init__.py` | Estimation module exports | Yes | No | Legacy exports | `src/scene_graph/estimation/__init__.py` | REFACTOR | Remove `GroundTruthEstimator` export. |
| `src/scene_graph/estimation/base.py` | BasePoseEstimator abstract class | Yes | No | No | `src/scene_graph/estimation/base.py` | KEEP | Authoritative estimator interface. |
| `src/scene_graph/estimation/identity.py` | Camera-local identity estimator | Yes | No | No | `src/scene_graph/estimation/identity.py` | KEEP | Fallback for local non-SLAM perception. |
| `src/scene_graph/estimation/rtabmap.py` | RTAB-Map SLAM pose estimator | Yes | No | No | `src/scene_graph/estimation/rtabmap.py` | KEEP | Authoritative SLAM backend integration. |
| `src/scene_graph/estimation/tf_buffer.py` | TF2 buffer pose lookup estimator | Yes | No | No | `src/scene_graph/estimation/tf_buffer.py` | KEEP | Authoritative ROS TF lookup integration. |
| `src/scene_graph/estimation/ground_truth.py`| TUM ground-truth file estimator | No | Yes (TUM) | Evaluation utility | `archive/tum/replay/ground_truth.py` | ARCHIVE | Evaluation benchmark pose estimator. |
| `src/scene_graph/geometry/__init__.py` | Geometry module exports | Yes | No | No | `src/scene_graph/geometry/__init__.py` | KEEP | Core geometry classes. |
| `src/scene_graph/geometry/camera.py` | CameraIntrinsics & DepthModel | Yes | No | No | `src/scene_graph/geometry/camera.py` | KEEP | Authoritative pinhole projection & frustum. |
| `src/scene_graph/geometry/downsampling.py` | Voxel grid point cloud downsampling | Yes | No | No | `src/scene_graph/geometry/downsampling.py` | KEEP | Real-time point cloud downsampling. |
| `src/scene_graph/geometry/ground_plane.py` | Ground plane estimation | Yes | No | No | `src/scene_graph/geometry/ground_plane.py` | KEEP | Global gravity & ground plane estimator. |
| `src/scene_graph/geometry/noise_model.py` | Stereo & depth sensor noise model | Yes | No | No | `src/scene_graph/geometry/noise_model.py` | KEEP | Authoritative depth variance model. |
| `src/scene_graph/geometry/plane.py` | RANSAC 3D plane detector | Yes | No | No | `src/scene_graph/geometry/plane.py` | KEEP | Geometric plane fitting. |
| `src/scene_graph/geometry/point_cloud.py` | Depth to 3D point cloud & OBB | Yes | No | No | `src/scene_graph/geometry/point_cloud.py` | KEEP | Metric 3D lifting & geometry estimation. |
| `src/scene_graph/geometry/provenance.py` | GeometrySource enum & tracking | Yes | No | No | `src/scene_graph/geometry/provenance.py` | KEEP | Observation vs predicted provenance. |
| `src/scene_graph/geometry/reference_frame.py`| RelationReferenceFrame SE(3) | Yes | No | No | `src/scene_graph/geometry/reference_frame.py` | KEEP | Global spatial reference frame convention. |
| `src/scene_graph/geometry/reference_frame_provider.py` | Sequence reference frame builder | Yes | No | Has dataset dict | `src/scene_graph/geometry/reference_frame_provider.py` | REFACTOR | Generalize dataset convention mapping. |
| `src/scene_graph/geometry/spatial_surface.py` | Spatial surface tracking | Yes | No | No | `src/scene_graph/geometry/spatial_surface.py` | KEEP | Support plane persistent caching. |
| `src/scene_graph/geometry/transforms.py` | SE(3) matrix / quaternion math | Yes | No | No | `src/scene_graph/geometry/transforms.py` | KEEP | Rigid body transformation utilities. |
| `src/scene_graph/graph/__init__.py` | Graph module exports | Yes | No | No | `src/scene_graph/graph/__init__.py` | KEEP | Graph package exports. |
| `src/scene_graph/graph/edge.py` | TemporalGraphEdge representation | Yes | No | No | `src/scene_graph/graph/edge.py` | KEEP | Directed spatial/temporal relation edge. |
| `src/scene_graph/graph/event.py` | GraphEvent lifecycle definitions | Yes | No | No | `src/scene_graph/graph/event.py` | KEEP | Node/edge appearance & change events. |
| `src/scene_graph/graph/graph_history.py` | Bounded circular event history | Yes | No | No | `src/scene_graph/graph/graph_history.py` | KEEP | History buffer for temporal reasoning. |
| `src/scene_graph/graph/node.py` | TemporalGraphNode representation | Yes | No | No | `src/scene_graph/graph/node.py` | KEEP | Persistent 3D object node entity. |
| `src/scene_graph/graph/participation_state.py`| Active vs dormant participation | Yes | No | No | `src/scene_graph/graph/participation_state.py` | KEEP | Lifecycle participation state flags. |
| `src/scene_graph/graph/query.py` | Spatial query engine | Yes | No | No | `src/scene_graph/graph/query.py` | KEEP | Predicate and entity query engine. |
| `src/scene_graph/graph/snapshot.py` | Static immutable graph snapshot | Yes | No | No | `src/scene_graph/graph/snapshot.py` | KEEP | Read-only graph state for query/VLM. |
| `src/scene_graph/graph/temporal_graph.py` | TemporalSceneGraph world model | Yes | No | No | `src/scene_graph/graph/temporal_graph.py` | KEEP | Authoritative semantic world model graph. |
| `src/scene_graph/ontology/__init__.py` | Ontology definitions | Yes | No | No | `src/scene_graph/ontology/__init__.py` | KEEP | Semantic ontology definitions. |
| `src/scene_graph/ontology/entity.py` | Entity taxonomy & role mappings | Yes | No | No | `src/scene_graph/ontology/entity.py` | KEEP | Entity classes and physical categories. |
| `src/scene_graph/ontology/relation.py` | Predicate taxonomy & hierarchy | Yes | No | No | `src/scene_graph/ontology/relation.py` | KEEP | Spatial relation taxonomy & opposites. |
| `src/scene_graph/perception/__init__.py` | Perception package exports | Yes | No | No | `src/scene_graph/perception/__init__.py` | KEEP | Perception interface exports. |
| `src/scene_graph/perception/observation.py` | 2D Detection Observation contract | Yes | No | No | `src/scene_graph/perception/observation.py` | KEEP | Authoritative detector output dataclass. |
| `src/scene_graph/perception/observation_source.py` | Abstract ObservationProducer API | Yes | No | No | `src/scene_graph/perception/observation_source.py` | KEEP | Protocol for detector vs replay loader. |
| `src/scene_graph/perception/stored_loader.py` | Chunked JSON observation replay | No | No | Replay utility | `archive/replay/stored_loader.py` | ARCHIVE | Offline observation cache loader. |
| `src/scene_graph/perception/yoloe_detector.py`| Open-vocab YOLOE segmentation | Yes | No | No | `src/scene_graph/perception/yoloe_detector.py` | KEEP | Authoritative live detector module. |
| `src/scene_graph/pipeline/__init__.py` | Pipeline module exports | Yes | No | No | `src/scene_graph/pipeline/__init__.py` | KEEP | Pipeline orchestration exports. |
| `src/scene_graph/pipeline/pipeline_core.py` | Core SceneGraphPipeline updater | Yes | No | No | `src/scene_graph/pipeline/pipeline_core.py` | KEEP | Authoritative single-frame update logic. |
| `src/scene_graph/pipeline/online_pipeline.py` | Synchronous online orchestration | Yes | No | No | `src/scene_graph/pipeline/online_pipeline.py` | KEEP | Live perception & graph orchestration. |
| `src/scene_graph/pipeline/async_pipeline.py` | Asynchronous worker & history buffer | Yes | No | No | `src/scene_graph/pipeline/async_pipeline.py` | KEEP | High-rate decoupled perception worker. |
| `src/scene_graph/pipeline/offline_pipeline.py`| Batch observation stream iterator | No | No | Replay adapter | `archive/replay/offline_pipeline.py` | ARCHIVE | Batch offline evaluation runner. |
| `src/scene_graph/relations/__init__.py` | Relation modules exports | Yes | No | No | `src/scene_graph/relations/__init__.py` | KEEP | Relation exports. |
| `src/scene_graph/relations/admissibility.py`| Geometric role admissibility filter | Yes | No | No | `src/scene_graph/relations/admissibility.py` | KEEP | Inadmissible relation pruning. |
| `src/scene_graph/relations/base.py` | BaseRelationModule abstract class | Yes | No | No | `src/scene_graph/relations/base.py` | KEEP | Spatial relation module interface. |
| `src/scene_graph/relations/candidate_generator.py` | Proximity candidate pair filtering | Yes | No | No | `src/scene_graph/relations/candidate_generator.py` | KEEP | Pairwise spatial candidate generation. |
| `src/scene_graph/relations/containment.py` | 3D AABB containment module | Yes | No | No | `src/scene_graph/relations/containment.py` | KEEP | INSIDE / CONTAINING inference. |
| `src/scene_graph/relations/context.py` | FrameContext geometry bundle | Yes | No | No | `src/scene_graph/relations/context.py` | KEEP | Contextual geometry per frame. |
| `src/scene_graph/relations/depth_order.py` | Camera-ray depth order module | Yes | No | No | `src/scene_graph/relations/depth_order.py` | KEEP | IN_FRONT_OF / BEHIND inference. |
| `src/scene_graph/relations/directional.py` | Global 3D directional relation module| Yes | No | No | `src/scene_graph/relations/directional.py` | KEEP | LEFT_OF, RIGHT_OF, ABOVE, BELOW. |
| `src/scene_graph/relations/distance.py` | 3D Euclidean distance module | Yes | No | No | `src/scene_graph/relations/distance.py` | KEEP | NEAR / FAR inference. |
| `src/scene_graph/relations/evidence.py` | RelationEvidence [0, 1] dataclass | Yes | No | No | `src/scene_graph/relations/evidence.py` | KEEP | Continuous belief evidence strength. |
| `src/scene_graph/relations/hierarchy.py` | Spatial hierarchy & tree building | Yes | No | No | `src/scene_graph/relations/hierarchy.py` | KEEP | Scene containment & support tree. |
| `src/scene_graph/relations/inverse_algebra.py`| Formal inverse predicate rules | Yes | No | No | `src/scene_graph/relations/inverse_algebra.py` | KEEP | Dual relation mapping & algebra. |
| `src/scene_graph/relations/occlusion.py` | 2D mask & depth occlusion module | Yes | No | No | `src/scene_graph/relations/occlusion.py` | KEEP | OCCLUDING / OCCLUDED_BY inference. |
| `src/scene_graph/relations/registry.py` | RelationRegistry module dispatcher | Yes | No | No | `src/scene_graph/relations/registry.py` | KEEP | Authoritative relation evaluation loop. |
| `src/scene_graph/relations/support.py` | Contact plane support module | Yes | No | No | `src/scene_graph/relations/support.py` | KEEP | ON / UNDER inference. |
| `src/scene_graph/temporal/__init__.py` | Temporal module exports | Yes | No | No | `src/scene_graph/temporal/__init__.py` | KEEP | State machine exports. |
| `src/scene_graph/temporal/object_state.py` | ObjectStateMachine (ACTIVE, etc.) | Yes | No | No | `src/scene_graph/temporal/object_state.py` | KEEP | Authoritative object lifecycle states. |
| `src/scene_graph/temporal/relation_state.py`| RelationStateMachine (Hysteresis) | Yes | No | No | `src/scene_graph/temporal/relation_state.py` | KEEP | Authoritative temporal relation states. |
| `src/scene_graph/tracking/__init__.py` | Tracking module exports | Yes | No | No | `src/scene_graph/tracking/__init__.py` | KEEP | Tracking exports. |
| `src/scene_graph/tracking/causal_tracker.py`| 3D Causal Kalman object tracker | Yes | No | No | `src/scene_graph/tracking/causal_tracker.py` | KEEP | Authoritative persistent object tracker. |
| `src/scene_graph/tracking/state.py` | TrackState (CONFIRMED, TENTATIVE) | Yes | No | No | `src/scene_graph/tracking/state.py` | KEEP | Tracker state representations. |
| `src/scene_graph/tracking/track.py` | Track entity with 3D state & bounds | Yes | No | No | `src/scene_graph/tracking/track.py` | KEEP | Persistent 3D object track dataclass. |
| `src/scene_graph/tracking/track_history.py`| Historical states of track | Yes | No | No | `src/scene_graph/tracking/track_history.py` | KEEP | Trajectory history storage. |
| `src/scene_graph/tracking/tracker_base.py` | BaseTracker abstract interface | Yes | No | No | `src/scene_graph/tracking/tracker_base.py` | KEEP | Protocol for object trackers. |
| `ros2_ws/src/d455_bridge/` (package) | Legacy Windows TCP -> ROS bridge | No | No | Legacy transport | `archive/legacy_transport/d455_windows_wsl/ros2_ws/src/d455_bridge/` | ARCHIVE | TCP development bridge superseded by Orin USB 3.x. |
| `ros2_ws/src/scene_graph_ros/launch/live_scene_graph.launch.py` | Authoritative live ROS 2 launch | Yes | No | Clean bridge arg | `ros2_ws/src/scene_graph_ros/launch/live_scene_graph.launch.py` | REFACTOR | Remove legacy TCP bridge dependency. |
| `ros2_ws/src/scene_graph_ros/launch/tum_scene_graph.launch.py` | TUM replay full launch | No | Yes (TUM) | Replay launch | `archive/tum/ros_launch/tum_scene_graph.launch.py` | ARCHIVE | Historical dataset replay launch. |
| `ros2_ws/src/scene_graph_ros/launch/tum_slam.launch.py` | TUM SLAM-only launch | No | Yes (TUM) | Replay launch | `archive/tum/ros_launch/tum_slam.launch.py` | ARCHIVE | Historical TUM SLAM launch. |
| `ros2_ws/src/scene_graph_ros/launch/tum_yolo.launch.py` | TUM YOLO-only launch | No | Yes (TUM) | Replay launch | `archive/tum/ros_launch/tum_yolo.launch.py` | ARCHIVE | Historical TUM perception launch. |
| `ros2_ws/src/scene_graph_ros/scene_graph_ros/tum_player.py` | TUM dataset ROS 2 player node | No | Yes (TUM) | Replay player | `archive/tum/replay/tum_player.py` | ARCHIVE | Replay node publishing to ROS clock. |
| `ros2_ws/src/scene_graph_ros/scene_graph_ros/config_loader.py` | Node config helper | Yes | No | No | `ros2_ws/src/scene_graph_ros/scene_graph_ros/config_loader.py` | KEEP | Clean configuration loader helper. |
| `ros2_ws/src/scene_graph_ros/scene_graph_ros/graph_publisher.py` | ROS 2 RViz Marker & State pub | Yes | No | No | `ros2_ws/src/scene_graph_ros/scene_graph_ros/graph_publisher.py` | KEEP | 3D scene graph visualization bridge. |
| `ros2_ws/src/scene_graph_ros/scene_graph_ros/live_2d_viewer.py` | OpenCV GUI 2D monitor | Yes | No | No | `ros2_ws/src/scene_graph_ros/scene_graph_ros/live_2d_viewer.py` | KEEP | Live 2D perception dashboard. |
| `ros2_ws/src/scene_graph_ros/scene_graph_ros/overlay_renderer.py`| 2D BBox & Track overlay drawer | Yes | No | No | `ros2_ws/src/scene_graph_ros/scene_graph_ros/overlay_renderer.py` | KEEP | Rendering utility for overlays. |
| `ros2_ws/src/scene_graph_ros/scene_graph_ros/ros_conversions.py` | ROS Msg <-> NumPy / SensorFrame | Yes | No | No | `ros2_ws/src/scene_graph_ros/scene_graph_ros/ros_conversions.py` | KEEP | Authoritative ROS 2 message adapter. |
| `ros2_ws/src/scene_graph_ros/scene_graph_ros/scene_graph_node.py` | Live ROS 2 perception node | Yes | No | Has 'tum' branch | `ros2_ws/src/scene_graph_ros/scene_graph_ros/scene_graph_node.py` | REFACTOR | Remove `if 'tum' in rgb_topic` branch. |
| `ros2_ws/src/scene_graph_ros/rviz/scene_graph.rviz` | RViz2 configuration | Yes | No | No | `ros2_ws/src/scene_graph_ros/rviz/scene_graph.rviz` | KEEP | Standard 3D visualization layout. |
| `tools/check_live_d455.py` | D455 Live Diagnostics Validator | Yes | No | No | `tools/diagnostics/check_live_d455.py` | MOVE | Authoritative hardware diagnostic tool. |
| `tools/sync_quality_report.py` | Timestamp sync statistics report | Yes | No | No | `tools/diagnostics/sync_quality_report.py` | MOVE | Timestamp synchronization report tool. |
| `tools/view_graph_logs.py` | Real-time terminal HUD dashboard | Yes | No | No | `tools/diagnostics/view_graph_logs.py` | MOVE | Terminal dashboard for `/scene_graph/state`. |
| `tools/record_d455.py` | RealSense D455 recorder | Yes | No | No | `tools/recording/record_d455.py` | MOVE | Direct camera sequence recorder. |
| `tools/compare_ros_regression.py` | Non-ROS vs ROS parity checker | Yes | No | No | `tools/evaluation/compare_ros_regression.py` | MOVE | Verification tool for ROS parity. |
| `tools/evaluate_stage3.py` | Trajectory ATE/RPE evaluator | Yes | No | Generic math | `tools/evaluation/evaluate_trajectory.py` | MOVE | Generalized trajectory benchmark utility. |
| `tools/class_distribution_report.py`| Class distribution report | Yes | No | No | `tools/maintenance/class_distribution_report.py` | MOVE | Dataset maintenance report tool. |
| `tools/d455_sender.py` | Windows pyrealsense2 TCP sender | No | No | Legacy transport | `archive/legacy_transport/d455_windows_wsl/d455_sender.py` | ARCHIVE | Windows TCP development bridge. |
| `tools/prepare_observations.py` | Offline YOLOE batch exporter | No | Yes (TUM) | Legacy tool | `archive/legacy_tools/prepare_observations.py` | ARCHIVE | Batch observation chunk creator. |
| `tools/run_online.py` | Local standalone TUM replay CLI | No | Yes (TUM) | Legacy tool | `archive/legacy_tools/run_online.py` | ARCHIVE | Hardcoded TUM replay script. |
| `tools/scene_graph_demo.py` | Streamlit TUM web demonstration | No | Yes (TUM) | Legacy tool | `archive/legacy_tools/scene_graph_demo.py` | ARCHIVE | Web GUI demo hardcoded to TUM source. |
| `scripts/run_live_d455.sh` | Main live bash runner script | Yes | No | No | `scripts/run_live_d455.sh` | KEEP | Authoritative shell entrypoint. |
| `scripts/run_tests_wsl.sh` | Pytest runner script | Yes | No | No | `scripts/run_tests.sh` | MOVE | General test suite runner. |
| `scripts/run_tum_benchmark.sh` | TUM benchmark replay script | No | Yes (TUM) | Legacy script | `archive/tum/scripts/run_tum_benchmark.sh` | ARCHIVE | TUM benchmark execution script. |
| `scripts/run_stage3_evaluation.sh`| Stage 3 evaluation shell script | No | Yes (mixed) | Legacy script | `archive/legacy_tools/run_stage3_evaluation.sh` | ARCHIVE | Stage 3 evaluation runner script. |
| `scripts/run_headless_benchmark.sh`| Headless replay runner script | No | Yes (mixed) | Legacy script | `archive/legacy_tools/run_headless_benchmark.sh` | ARCHIVE | Headless replay execution script. |
| `run_demo.sh` | Root demo runner | No | Yes (TUM) | Legacy script | `archive/tum/run_demo.sh` | ARCHIVE | Shell script launching TUM demo. |
| `run_live_d455.sh` | Root live forwarder | Yes | No | No | `run_live_d455.sh` | KEEP | Convenient forwarder to `scripts/run_live_d455.sh`. |
| `docs/architecture.md` | System architecture doc | Yes | No | No | `docs/architecture/system.md` | MOVE | Authoritative architecture document. |
| `docs/data.md` | Data structures & contract doc | Yes | No | No | `docs/architecture/data_flow.md` | MOVE | Data contract document. |
| `docs/scene_graph_model.md` | Scene graph model specification | Yes | No | No | `docs/architecture/scene_graph.md` | MOVE | Scene graph model document. |
| `docs/geometry.md` | Geometry specification | Yes | No | No | `docs/architecture/geometry.md` | MOVE | 3D geometry specification. |
| `docs/relations.md` | Spatial relations specification | Yes | No | No | `docs/architecture/relations.md` | MOVE | Relations specification. |
| `docs/temporal.md` | Temporal state specification | Yes | No | No | `docs/architecture/temporal.md` | MOVE | Temporal state specification. |
| `docs/evaluation_protocol.md` | Evaluation protocol doc | Yes | No | No | `docs/evaluation/protocol.md` | MOVE | Evaluation benchmark protocol. |
| `docs/d455_windows_wsl_bridge.md` | Windows/WSL TCP bridge doc | No | No | Legacy transport doc | `archive/legacy_transport/d455_windows_wsl/d455_windows_wsl_bridge.md` | ARCHIVE | Documentation for historical transport. |
| `docs/tum_data.md` | TUM dataset notes | No | Yes (TUM) | Dataset doc | `archive/tum/docs/tum_data.md` | ARCHIVE | Documentation for historical TUM dataset. |
| `docs/stage1/*` (9 files) | Stage 1 architecture & testing docs | No | No | Historical stage docs | `archive/historical_docs/stage1/` | ARCHIVE | Replaced by clean `docs/stages/` docs. |
| `results/stage3/tum_fr1_desk/` | Historical TUM Stage 3 results | No | Yes (TUM) | Historical artifacts | `archive/historical_results/tum_fr1_desk/` | ARCHIVE | Benchmark outputs. |
| `results/stage3/my_desk_sequence/`| Historical my_desk Stage 3 results | No | Yes (my_desk) | Historical artifacts | `archive/historical_results/my_desk_sequence/` | ARCHIVE | Experiment outputs. |
| `tests/unit/test_camera.py` | Pinhole camera geometry tests | Yes | No | No | `tests/unit/geometry/test_camera.py` | MOVE | Unit tests for camera geometry. |
| `tests/unit/test_camera_geometry.py`| Camera frustum & box clipping | Yes | No | No | `tests/unit/geometry/test_camera_geometry.py` | MOVE | Unit tests for camera frustum. |
| `tests/unit/test_depth_noise_model.py`| Sensor depth noise model | Yes | No | No | `tests/unit/geometry/test_depth_noise_model.py` | MOVE | Unit tests for depth noise. |
| `tests/unit/test_geometry.py` | Point cloud geometry tests | Yes | No | No | `tests/unit/geometry/test_geometry.py` | MOVE | Unit tests for 3D point cloud. |
| `tests/unit/test_persistent_surfaces.py`| Surface caching tests | Yes | No | No | `tests/unit/geometry/test_persistent_surfaces.py` | MOVE | Unit tests for spatial surfaces. |
| `tests/unit/test_point_cloud.py` | Point cloud utilities tests | Yes | No | No | `tests/unit/geometry/test_point_cloud.py` | MOVE | Unit tests for point cloud functions. |
| `tests/unit/test_reference_frame_modes.py`| RelationReferenceFrame tests | Yes | No | No | `tests/unit/geometry/test_reference_frame_modes.py` | MOVE | Unit tests for reference frames. |
| `tests/unit/test_sensor_noise_covariance.py`| Sensor noise covariance | Yes | No | No | `tests/unit/geometry/test_sensor_noise_covariance.py` | MOVE | Unit tests for covariance models. |
| `tests/unit/test_spatial_caching.py`| Spatial point cloud caching | Yes | No | No | `tests/unit/geometry/test_spatial_caching.py` | MOVE | Unit tests for spatial cache. |
| `tests/unit/test_detector.py` | Observation producer tests | Yes | No | StoredLoader ref | `tests/unit/perception/test_detector.py` | REFACTOR | Decouple from stored_loader. |
| `tests/unit/test_observation.py` | Observation dataclass tests | Yes | No | No | `tests/unit/perception/test_observation.py` | MOVE | Unit tests for observation. |
| `tests/unit/test_open_vocabulary_perception.py`| YOLOE open-vocab tests | Yes | No | No | `tests/unit/perception/test_open_vocabulary_perception.py` | MOVE | Unit tests for YOLOE detector. |
| `tests/unit/test_causal_tracker.py` | CausalTracker Kalman tests | Yes | No | No | `tests/unit/tracking/test_causal_tracker.py` | MOVE | Unit tests for tracker. |
| `tests/unit/test_class_flip_tracking.py`| Semantic flip robustness tests | Yes | No | No | `tests/unit/tracking/test_class_flip_tracking.py` | MOVE | Unit tests for class flip. |
| `tests/unit/test_decoupled_tracking.py`| Decoupled 3D state tests | Yes | No | No | `tests/unit/tracking/test_decoupled_tracking.py` | MOVE | Unit tests for decoupled tracking. |
| `tests/unit/test_partial_observability.py`| Dropout / occlusion tests | Yes | No | No | `tests/unit/tracking/test_partial_observability.py` | MOVE | Unit tests for partial observability. |
| `tests/unit/test_temporal_tracking_robustness.py`| Tracker lifecycle tests | Yes | No | No | `tests/unit/tracking/test_temporal_tracking_robustness.py` | MOVE | Unit tests for track lifecycle. |
| `tests/unit/test_candidate_generator.py`| Pairwise relation candidates | Yes | No | No | `tests/unit/relations/test_candidate_generator.py` | MOVE | Unit tests for candidate pairs. |
| `tests/unit/test_geometric_admissibility.py`| Admissibility filter tests | Yes | No | No | `tests/unit/relations/test_geometric_admissibility.py` | MOVE | Unit tests for admissibility rules. |
| `tests/unit/test_hierarchy.py` | Spatial hierarchy tests | Yes | No | No | `tests/unit/relations/test_hierarchy.py` | MOVE | Unit tests for hierarchy trees. |
| `tests/unit/test_ontology.py` | Entity & relation taxonomy | Yes | No | No | `tests/unit/relations/test_ontology.py` | MOVE | Unit tests for ontology. |
| `tests/unit/test_relation_evidence_strength.py`| Continuous evidence tests | Yes | No | No | `tests/unit/relations/test_relation_evidence_strength.py` | MOVE | Unit tests for evidence strength. |
| `tests/unit/test_relations.py` | Core relation modules tests | Yes | No | No | `tests/unit/relations/test_relations.py` | MOVE | Unit tests for all relation modules. |
| `tests/unit/test_bounded_history.py`| Bounded memory history tests | Yes | No | No | `tests/unit/temporal/test_bounded_history.py` | MOVE | Unit tests for memory bounded history. |
| `tests/unit/test_temporal_object_state.py`| ObjectStateMachine tests | Yes | No | No | `tests/unit/temporal/test_temporal_object_state.py` | MOVE | Unit tests for object state machine. |
| `tests/unit/test_temporal_relations_lifecycle.py`| Relation state lifecycle | Yes | No | No | `tests/unit/temporal/test_temporal_relations_lifecycle.py` | MOVE | Unit tests for relation state machine. |
| `tests/unit/test_time_and_localization_semantics.py`| Localization semantics | Yes | No | No | `tests/unit/temporal/test_time_and_localization_semantics.py` | MOVE | Unit tests for time semantics. |
| `tests/unit/test_snapshot.py` | Immutable snapshot tests | Yes | No | No | `tests/unit/graph/test_snapshot.py` | MOVE | Unit tests for graph snapshot. |
| `tests/unit/test_scene_graph_demo.py`| Demo helper tests | No | Yes (TUM) | Legacy test | `archive/legacy_tests/test_scene_graph_demo.py` | ARCHIVE | Streamlit demo unit test. |
| `tests/unit/test_data_contracts.py` | SensorFrame / FramePacket tests | Yes | No | No | `tests/unit/data/test_data_contracts.py` | MOVE | Unit tests for data contracts. |
| `tests/unit/test_localization_and_timestamps.py`| Timestamp domain tests | Yes | No | No | `tests/unit/data/test_localization_and_timestamps.py` | MOVE | Unit tests for timestamp domains. |
| `tests/unit/test_sensor_contracts.py`| Metric depth scale tests | Yes | No | No | `tests/unit/data/test_sensor_contracts.py` | MOVE | Unit tests for metric depth contracts. |
| `tests/unit/test_overlay_renderer.py`| OpenCV overlay drawing tests | Yes | No | No | `tests/unit/ros/test_overlay_renderer.py` | MOVE | Unit tests for 2D overlay renderer. |
| `tests/unit/test_ros_conversions.py` | ROS 2 message conversion tests | Yes | No | No | `tests/unit/ros/test_ros_conversions.py` | MOVE | Unit tests for ROS converters. |
| `tests/stage1/unit/test_buffer_policy.py`| Latest-frame queue policy | Yes | No | No | `tests/unit/sensor/test_buffer_policy.py` | MOVE | Sensor queue drop policy tests. |
| `tests/stage1/unit/test_calibration.py`| Camera calibration roundtrip | Yes | No | No | `tests/unit/sensor/test_calibration.py` | MOVE | Intrinsics & distortion model tests. |
| `tests/stage1/unit/test_clock_mapping.py`| Clock mapping & drift tests | Yes | No | No | `tests/unit/sensor/test_clock_mapping.py` | MOVE | Clock synchronization filter tests. |
| `tests/stage1/unit/test_depth_scale.py`| D455 depth scale bounds | Yes | No | No | `tests/unit/sensor/test_depth_scale.py` | MOVE | Depth scale validation tests. |
| `tests/stage1/unit/test_depth_scale_propagation.py`| Runtime depth scale prop | Yes | No | No | `tests/unit/sensor/test_depth_scale_propagation.py` | MOVE | Runtime depth scale propagation tests. |
| `tests/stage1/unit/test_imu_contract.py`| IMU telemetry timestamp tests | Yes | No | No | `tests/unit/sensor/test_imu_contract.py` | MOVE | IMU contract tests. |
| `tests/stage1/unit/test_packet_validation.py`| Sensor packet validation | Yes | No | No | `tests/unit/sensor/test_packet_validation.py` | MOVE | Ingestion payload validation tests. |
| `tests/stage1/unit/test_regression_previous_failures.py`| Regression test suite | Yes | No | No | `tests/regression/test_sensor_regression.py` | MOVE | Known sensor failure protection tests. |
| `tests/stage1/unit/test_sensor_contract.py`| SensorFrame contract tests | Yes | No | No | `tests/unit/sensor/test_sensor_contract.py` | MOVE | SensorFrame invariants tests. |
| `tests/stage1/unit/test_session_provenance.py`| Session ID & provenance | Yes | No | No | `tests/unit/data/test_session_provenance.py` | MOVE | Provenance metadata tests. |
| `tests/stage1/unit/test_synchronization.py`| RGB-Depth sync tolerance | Yes | No | No | `tests/unit/data/test_synchronization.py` | MOVE | Timestamp sync tolerance tests. |
| `tests/stage1/unit/test_timestamp_contract.py`| Monotonic & system time | Yes | No | No | `tests/unit/data/test_timestamp_contract.py` | MOVE | Timestamp class contract tests. |
| `tests/stage1/unit/test_transport_protocol.py`| Protocol v2 framing tests | No | No | Legacy transport test | `archive/legacy_transport/d455_windows_wsl/tests/test_transport_protocol.py` | ARCHIVE | TCP protocol v2 framing tests. |
| `tests/stage1/integration/test_d455_sender_receiver.py`| TCP bridge loopback test | No | No | Legacy transport test | `archive/legacy_transport/d455_windows_wsl/tests/test_d455_sender_receiver.py` | ARCHIVE | Windows/WSL TCP loopback test. |
| `tests/stage1/integration/test_imu_ingestion.py`| ROS IMU topic ingestion | Yes | No | No | `tests/integration/test_imu_ingestion.py` | MOVE | IMU topic ingestion test. |
| `tests/stage1/integration/test_reconnect.py`| TCP socket reconnect test | No | No | Legacy transport test | `archive/legacy_transport/d455_windows_wsl/tests/test_reconnect.py` | ARCHIVE | TCP socket reconnect test. |
| `tests/stage1/integration/test_record_replay_parity.py`| Parity of recorded frames | Yes | No | No | `tests/replay/test_record_replay_parity.py` | MOVE | Ingestion vs replay parity test. |
| `tests/stage1/integration/test_rgb_depth_ingestion.py`| ROS RGB-D topic ingestion | Yes | No | No | `tests/integration/test_rgb_depth_ingestion.py` | MOVE | ROS RGB-D message delivery test. |
| `tests/stage1/integration/test_ros_ingestion.py`| ROS conversions integration | Yes | No | No | `tests/integration/test_ros_ingestion.py` | MOVE | ROS message to SensorFrame test. |
| `tests/stage1/system/hardware/test_d455_startup.py`| Physical camera live test | Yes (HW) | No | No | `tests/hardware/d455/test_startup.py` | MOVE | Physical D455 hardware connection test. |
| `tests/stage1/system/test_latency_budget.py`| Ingestion latency budget | Yes | No | No | `tests/integration/test_latency_budget.py` | MOVE | Ingestion throughput & latency test. |
| `tests/stage3/test_stage3_pose_estimation.py`| Pose estimators test suite | Yes | No | Sections A-F core | `tests/unit/estimation/test_pose_estimation.py` | REFACTOR | Relocate core tests; Section G to replay. |
| `tests/integration/test_causality.py` | Pipeline causality tests | Yes | No | No | `tests/integration/test_causality.py` | KEEP | Ingestion causality integration test. |
| `tests/integration/test_online_pipeline.py`| Online pipeline test | Yes | No | No | `tests/integration/test_online_pipeline.py` | KEEP | End-to-end online pipeline test. |
| `tests/regression/test_full_pipeline_regression.py`| 10 Master invariants tests | Yes | No | No | `tests/regression/test_full_pipeline_regression.py` | KEEP | Core architectural regression tests. |
| `tests/regression/test_geometry_fixtures.py`| Synthetic geometry tests | Yes | No | No | `tests/regression/test_geometry_fixtures.py` | KEEP | Geometry fixture verification tests. |
| `tests/regression/test_master_diagnostics_suite.py`| 13 System requirements tests| Yes | No | No | `tests/regression/test_master_diagnostics_suite.py` | KEEP | System requirements regression tests. |

---

## 3. Discovered Duplicates, Divergences, and Contaminations

1. **Dataset-Specific Contamination in Core / Node**:
   - `ros2_ws/src/scene_graph_ros/scene_graph_ros/scene_graph_node.py` (Line 296):
     ```python
     dataset_type = "tum" if "tum" in self.rgb_topic.lower() else "realsense"
     self.ref_frame_provider = ReferenceFrameProvider(dataset_type=dataset_type)
     ```
     *Fix*: Remove dataset inspection string matching. Default to `"realsense"` or configure via ROS parameter `dataset_type` (defaulting to `"realsense"`).
   - `src/scene_graph/data/__init__.py`:
     Directly exported `TUMLoader`, `TUMReplaySource`, `load_tum_rgb`, `load_tum_depth`, `load_tum_groundtruth`.
     *Fix*: Remove TUM exports from core `data/__init__.py`.
   - `src/scene_graph/estimation/__init__.py`:
     Directly exported `GroundTruthEstimator` (which in turn imported from `tum_loader`).
     *Fix*: Remove `GroundTruthEstimator` from core `estimation/__init__.py`.

2. **Legacy Windows/WSL Transport Bridge**:
   - `tools/d455_sender.py` (pyrealsense2 Windows TCP streamer)
   - `ros2_ws/src/d455_bridge/d455_bridge/d455_receiver.py` (WSL TCP receiver)
   - `live_scene_graph.launch.py` had default `use_bridge:=true`.
   - *Fix*: Move sender and receiver into `archive/legacy_transport/d455_windows_wsl/`. Update `live_scene_graph.launch.py` to default `use_bridge:=false` (as RealSense will run directly on Orin via standard ROS 2 camera drivers).

3. **Duplicated Replay / Demonstration Pipelines**:
   - `tools/run_online.py`: Standalone CLI runner hardcoded to `TUMReplaySource`.
   - `tools/scene_graph_demo.py`: Streamlit web UI hardcoded to `TUMReplaySource`.
   - `src/scene_graph/pipeline/offline_pipeline.py`: Redundant wrapper around `SceneGraphPipeline`.
   - `src/scene_graph/perception/stored_loader.py`: Pre-computed JSON chunk observation loader.
   - *Fix*: Move `run_online.py`, `scene_graph_demo.py`, `prepare_observations.py` to `archive/legacy_tools/`. Move `stored_loader.py` and `offline_pipeline.py` to `archive/replay/`.

4. **Multiple Scattered Test Directories**:
   - Tests were fragmented across `tests/stage1/unit`, `tests/stage1/integration`, `tests/stage1/system`, `tests/stage3/`, `tests/unit/`, `tests/integration/`, `tests/regression/`.
   - *Fix*: Reorganize into a clean functional tree: `unit/{data, sensor, geometry, perception, tracking, relations, temporal, graph, estimation, ros}`, `integration/`, `replay/`, `regression/`, `hardware/d455/`.
