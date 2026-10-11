"""Generic SceneGraph ROS Node."""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any

def _ensure_paths():
    repo_root = None
    current = Path(__file__).resolve().parent
    while current != current.parent:
        candidate_src = current / "src"
        if (candidate_src / "scene_graph").exists():
            src_str = str(candidate_src)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            repo_root = current
            break
        current = current.parent
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = []
    custom_venv = os.environ.get("SCENE_GRAPH_VENV")
    if custom_venv:
        candidates.append(Path(custom_venv))
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidates.append(Path(venv))
    if repo_root:
        candidates.extend([repo_root / ".venv", repo_root / "myenv", repo_root / "venv"])
    candidates.extend([Path.home() / ".venv", Path.home() / "myenv", Path.home() / "venv"])
    ver_list = [py_ver, "python3.10", "python3.11", "python3.12", "python3.9", "python3"]
    for c in candidates:
        if not c.exists():
            continue
        for ver in ver_list:
            sp = c / "lib" / ver / "site-packages"
            if sp.exists() and str(sp) not in sys.path:
                sys.path.insert(0, str(sp))
                break
_ensure_paths()

import csv
import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
import cv2
import message_filters
import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, Imu

try:
    from rtabmap_msgs.msg import Info as RtabmapInfo, OdomInfo as RtabmapOdomInfo
    from nav_msgs.msg import Odometry
    HAS_RTABMAP_MSGS = True
except ImportError:
    HAS_RTABMAP_MSGS = False
    RtabmapInfo = None
    RtabmapOdomInfo = None
    Odometry = None

from scene_graph.config import SceneGraphConfig
from scene_graph.data.frame_packet import FramePacket, IMUSample, LocalizationMode
from scene_graph.data.pose_estimate import EstimatorState, PoseEstimate
from scene_graph.data.sensor_frame import SensorFrame, StreamStatus
from scene_graph.data.timestamp import Timestamp, TimestampDomain
from scene_graph.estimation.base import BasePoseEstimator
from scene_graph.estimation.identity import IdentityEstimator
from scene_graph.estimation.rtabmap import RTABMapEstimator
from scene_graph.estimation.tf_buffer import TFBufferEstimator
from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.geometry.point_cloud import GeometryStatus
from scene_graph.geometry.reference_frame import RelationReferenceFrame
from scene_graph.geometry.reference_frame_provider import ReferenceFrameProvider
from scene_graph.geometry.transforms import matrix_to_quaternion
from scene_graph.graph.snapshot import create_snapshot
from scene_graph.pipeline.online_pipeline import OnlinePipeline
from scene_graph.tracking.track import TrackState
from scene_graph_ros.config_loader import load_scene_graph_config
from scene_graph_ros.graph_publisher import GraphPublisher
from scene_graph_ros.overlay_renderer import (
    render_detections_overlay,
    render_tracks_overlay,
)
from scene_graph_ros.ros_conversions import (
    camera_info_to_intrinsics,
    imu_msg_to_sample,
    ros_image_to_numpy,
)

@dataclass
class PendingFrame:
    """Container for synchronized raw ROS messages awaiting worker processing."""
    rgb_msg: Image
    depth_msg: Image
    camera_info_msg: CameraInfo | None
    arrival_time: float

class SceneGraphROSNode(Node):
    """ROS 2 node orchestrating online 3D scene graph generation."""
    def __init__(self):
        super().__init__("scene_graph_node")
        self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("sensor_frame", "camera_color_optical_frame")
        self.declare_parameter("rgb_depth_max_dt", 0.02)
        self.declare_parameter("pose_max_dt", 0.05)
        self.declare_parameter("use_imu", False)
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("config_path", "configs/runtime/live_d455.yaml")
        self.declare_parameter("depth_scale", 1000.0)
        self.declare_parameter("debug_frame_packet_only", False)
        self.declare_parameter("state_topic", "/scene_graph/state")
        self.declare_parameter("markers_topic", "/scene_graph/markers")
        self.declare_parameter("object_cloud_topic", "/scene_graph/object_cloud")
        self.declare_parameter("scene_cloud_topic", "/scene_graph/scene_cloud")
        self.declare_parameter("publish_scene_cloud", False)
        self.declare_parameter("scene_cloud_stride", 4)
        self.declare_parameter("map_cloud_topic", "/scene_graph/map_cloud")
        self.declare_parameter("map_voxel_size_m", 0.02)
        self.declare_parameter("map_max_points", 500000)
        self.declare_parameter("map_cloud_stride", 4)
        self.declare_parameter("overlay_detections_topic", "/scene_graph/overlay_detections")
        self.declare_parameter("overlay_tracks_topic", "/scene_graph/overlay_tracks")
        self.declare_parameter("queue_size", 2)
        self.declare_parameter("drop_old_frames", True)
        self.declare_parameter("localization_mode", "slam")
        self.declare_parameter("async_mode", True)
        self.declare_parameter("min_hits", -1)
        self.declare_parameter("max_missing_seconds", -1.0)
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", False)
        self.declare_parameter("use_latest_tf", True)
        self.declare_parameter("slam_pose_max_age", 0.08)
        self.declare_parameter("telemetry_log_path", "")
        self.declare_parameter("export_dir", "")
        self.declare_parameter("telemetry_max_age", 0.25)
        self.declare_parameter("rtabmap_odom_topic", "/rtabmap/odom")
        self.declare_parameter("rtabmap_odom_info_topic", "/rtabmap/odom_info")
        self.declare_parameter("rtabmap_info_topic", "/rtabmap/info")

        self.rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.world_frame = self.get_parameter("world_frame").get_parameter_value().string_value
        self.sensor_frame = self.get_parameter("sensor_frame").get_parameter_value().string_value
        self.rgb_depth_max_dt = self.get_parameter("rgb_depth_max_dt").get_parameter_value().double_value
        self.pose_max_dt = self.get_parameter("pose_max_dt").get_parameter_value().double_value
        self.use_imu = self.get_parameter("use_imu").get_parameter_value().bool_value
        self.imu_topic = self.get_parameter("imu_topic").get_parameter_value().string_value
        self.config_path = self.get_parameter("config_path").get_parameter_value().string_value
        self.param_depth_scale = self.get_parameter("depth_scale").get_parameter_value().double_value
        self.debug_frame_packet_only = self.get_parameter("debug_frame_packet_only").get_parameter_value().bool_value
        self.state_topic = self.get_parameter("state_topic").get_parameter_value().string_value
        self.markers_topic = self.get_parameter("markers_topic").get_parameter_value().string_value
        self.object_cloud_topic = self.get_parameter("object_cloud_topic").get_parameter_value().string_value
        self.scene_cloud_topic = self.get_parameter("scene_cloud_topic").get_parameter_value().string_value
        self.publish_scene_cloud = self.get_parameter("publish_scene_cloud").get_parameter_value().bool_value
        self.scene_cloud_stride = self.get_parameter("scene_cloud_stride").get_parameter_value().integer_value
        self.map_cloud_topic = self.get_parameter("map_cloud_topic").get_parameter_value().string_value
        self.map_voxel_size_m = self.get_parameter("map_voxel_size_m").get_parameter_value().double_value
        self.map_max_points = self.get_parameter("map_max_points").get_parameter_value().integer_value
        self.map_cloud_stride = self.get_parameter("map_cloud_stride").get_parameter_value().integer_value
        self.overlay_detections_topic = self.get_parameter("overlay_detections_topic").get_parameter_value().string_value
        self.overlay_tracks_topic = self.get_parameter("overlay_tracks_topic").get_parameter_value().string_value
        self.queue_size = self.get_parameter("queue_size").get_parameter_value().integer_value

        raw_drop = self.get_parameter("drop_old_frames").value
        self.drop_old_frames = raw_drop.strip().lower() in ("true", "1", "yes") if isinstance(raw_drop, str) else bool(raw_drop)
        self.localization_mode = self.get_parameter("localization_mode").get_parameter_value().string_value.lower()
        raw_async = self.get_parameter("async_mode").value
        self.async_mode = raw_async.strip().lower() in ("true", "1", "yes") if isinstance(raw_async, str) else bool(raw_async)
        self.param_min_hits = self.get_parameter("min_hits").get_parameter_value().integer_value
        self.param_max_missing_seconds = self.get_parameter("max_missing_seconds").get_parameter_value().double_value
        raw_tf = self.get_parameter("use_latest_tf").value
        self.use_latest_tf = raw_tf.strip().lower() in ("true", "1", "yes") if isinstance(raw_tf, str) else bool(raw_tf)
        self.slam_pose_max_age = self.get_parameter("slam_pose_max_age").get_parameter_value().double_value
        self.telemetry_log_path = self.get_parameter("telemetry_log_path").get_parameter_value().string_value
        self.export_dir = self.get_parameter("export_dir").get_parameter_value().string_value
        self.telemetry_max_age = self.get_parameter("telemetry_max_age").get_parameter_value().double_value
        self.rtabmap_odom_topic = self.get_parameter("rtabmap_odom_topic").get_parameter_value().string_value
        self.rtabmap_odom_info_topic = self.get_parameter("rtabmap_odom_info_topic").get_parameter_value().string_value
        self.rtabmap_info_topic = self.get_parameter("rtabmap_info_topic").get_parameter_value().string_value

        self.get_logger().info(f"Loading SceneGraph configuration from {self.config_path}")
        self.config: SceneGraphConfig = load_scene_graph_config(self.config_path)

        if self.param_min_hits > 0 and self.config.tracking and self.config.tracking.confirmation:
            self.config.tracking.confirmation.min_hits = self.param_min_hits
            self.get_logger().info(f"Applied parameter override: tracking.confirmation.min_hits={self.param_min_hits}")

        if self.param_max_missing_seconds > 0 and self.config.tracking and self.config.tracking.occlusion:
            self.config.tracking.occlusion.max_missing_seconds = self.param_max_missing_seconds
            self.get_logger().info(f"Applied parameter override: tracking.occlusion.max_missing_seconds={self.param_max_missing_seconds:.1f}s")

        depth_scale = self.param_depth_scale
        if self.config.depth is not None and self.config.depth.scale > 0:
            depth_scale = self.config.depth.scale
        self.depth_scale = float(depth_scale)
        self.depth_model = DepthModel(scale=self.depth_scale)
        self.get_logger().info(f"Depth adapter configured: {self.depth_scale:.1f} raw units per meter")

        tf_mode_str = (
            f"SLAM/latest-TF mode (Time(0), max_age={self.slam_pose_max_age:.1f}s)"
            if self.use_latest_tf
            else f"exact-timestamp mode (pose_max_dt={self.pose_max_dt*1000:.0f}ms)"
        )
        self.get_logger().info(f"Localization: mode={self.localization_mode}, TF-lookup={tf_mode_str}")

        if not self.debug_frame_packet_only:
            self.get_logger().info(f"Initializing OnlinePipeline (async_mode={self.async_mode})...")
            self.pipeline = OnlinePipeline(self.config, async_mode=self.async_mode)
            if hasattr(self.pipeline, "detector") and hasattr(self.pipeline.detector, "get_model_info"):
                m_info = self.pipeline.detector.get_model_info()
                self.get_logger().info(
                    f"Detector Provenance | checkpoint: {m_info['model_path']} (sha256: {m_info['model_sha256'][:8]}...) "
                    f"| ultralytics: {m_info['ultralytics_version']} | mode: {m_info['mode']} "
                    f"| classes ({len(m_info.get('current_text_prompts') or [])}): {m_info.get('current_text_prompts')} "
                    f"| conf: {m_info['confidence_threshold']} | device: {m_info['device']}"
                )
            self.get_logger().info("OnlinePipeline ready.")
        else:
            self.pipeline = None
            self.get_logger().warn("Running in debug_frame_packet_only mode: Core pipeline bypassed.")

        self.graph_publisher = GraphPublisher(
            node=self,
            state_topic=self.state_topic,
            markers_topic=self.markers_topic,
            object_cloud_topic=self.object_cloud_topic,
            scene_cloud_topic=self.scene_cloud_topic,
            publish_scene_cloud=self.publish_scene_cloud,
            scene_cloud_stride=self.scene_cloud_stride,
            map_cloud_topic=self.map_cloud_topic,
            map_voxel_size_m=self.map_voxel_size_m,
            map_max_points=self.map_max_points,
            map_cloud_stride=self.map_cloud_stride,
            overlay_detections_topic=self.overlay_detections_topic,
            overlay_tracks_topic=self.overlay_tracks_topic,
            world_frame=self.world_frame,
        )

        self.latest_camera_info: CameraInfo | None = None
        self.latest_intrinsics: CameraIntrinsics | None = None
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.tf_buffer = tf2_ros.Buffer(node=self)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        reference_config = self.config.reference_frame
        ref_dataset = getattr(reference_config, "dataset_type", "d455") if reference_config else "d455"
        self.ref_frame_provider = ReferenceFrameProvider(dataset_type=ref_dataset)

        if self.localization_mode == "camera_local":
            self.pose_estimator: BasePoseEstimator = IdentityEstimator(frame_id=self.sensor_frame)
        elif self.localization_mode in ("world", "tf"):
            self.pose_estimator = TFBufferEstimator(
                tf_buffer=self.tf_buffer,
                world_frame=self.world_frame,
                sensor_frame=self.sensor_frame,
                pose_max_dt=self.pose_max_dt,
                slam_pose_max_age=self.slam_pose_max_age,
                allow_causal_fallback=self.use_latest_tf,
                backend_name="tf_buffer",
            )
        elif self.localization_mode in ("slam", "rtabmap"):
            self.pose_estimator = RTABMapEstimator(
                tf_buffer=self.tf_buffer,
                world_frame=self.world_frame,
                sensor_frame=self.sensor_frame,
                pose_max_dt=self.pose_max_dt,
                slam_pose_max_age=self.slam_pose_max_age,
                telemetry_max_age=self.telemetry_max_age,
                allow_causal_fallback=self.use_latest_tf,
                backend_name="rtabmap",
            )
        else:
            raise ValueError(
                f"Unsupported localization_mode '{self.localization_mode}'. "
                "Expected 'slam', 'rtabmap', 'world', 'tf', or 'camera_local'."
            )

        self.odom_info_sub = None
        self.info_sub = None
        self.odom_sub = None

        if HAS_RTABMAP_MSGS and isinstance(self.pose_estimator, RTABMapEstimator):
            self.odom_info_sub = self.create_subscription(
                RtabmapOdomInfo,
                self.rtabmap_odom_info_topic,
                self._odom_info_callback,
                10,
            )
            self.info_sub = self.create_subscription(
                RtabmapInfo,
                self.rtabmap_info_topic,
                self._info_callback,
                10,
            )
            self.odom_sub = self.create_subscription(
                Odometry,
                self.rtabmap_odom_topic,
                self._odom_callback,
                10,
            )
            self.get_logger().info(
                f"RTAB-Map telemetry subscriptions registered:\n"
                f"  OdomInfo: {self.rtabmap_odom_info_topic}\n"
                f"  Info: {self.rtabmap_info_topic}\n"
                f"  Odom: {self.rtabmap_odom_topic}"
            )

        self._telemetry_file = None
        self._telemetry_writer = None
        self._trajectory_file = None

        self._export_path: Path | None = None
        self._detections_file = None
        self._geometry_file = None
        self._track_events_file = None
        self._relation_events_file = None
        self._video_writer = None
        self._gallery_saved = 0

        self.metrics = {
            "input_rgbd_pairs": 0,
            "synchronized_frames": 0,
            "frames_with_valid_slam_pose": 0,
            "frames_with_unlocalized_or_missing_pose": 0,
            "detector_invocations": 0,
            "total_detections": 0,
            "valid_3d_geometries": 0,
            "unlocalized_geometries": 0,
            "invalid_or_out_of_range_depth_geometries": 0,
            "primary_geometry_outcomes": {
                "localized_valid_geometry": 0,
                "localized_insufficient_depth": 0,
                "unlocalized_valid_geometry": 0,
                "unlocalized_insufficient_depth": 0,
            },
            "tracker_updates": 0,
            "confirmed_tracks": 0,
            "graph_updates": 0,
            "active_nodes_final": 0,
            "active_edges_final": 0,
            "emitted_relations": 0,
            "dropped_frames": 0,
            "discard_reasons": {
                "sync_skew_exceeded": 0,
                "queue_overflow": 0,
                "missing_intrinsics": 0,
                "unlocalized_pose": 0,
                "no_depth_or_points": 0,
            },
            "latencies_ms": {
                "estimation": [],
                "inference": [],
                "total": [],
            },
        }

        if self.export_dir:
            self._export_path = Path(self.export_dir)
            (self._export_path / "slam").mkdir(parents=True, exist_ok=True)
            (self._export_path / "perception").mkdir(parents=True, exist_ok=True)
            (self._export_path / "geometry").mkdir(parents=True, exist_ok=True)
            (self._export_path / "tracking").mkdir(parents=True, exist_ok=True)
            (self._export_path / "scene_graph" / "snapshots").mkdir(parents=True, exist_ok=True)
            (self._export_path / "visualizations").mkdir(parents=True, exist_ok=True)

            self._detections_file = open(self._export_path / "perception" / "detections.jsonl", "w", encoding="utf-8")
            self._geometry_file = open(self._export_path / "geometry" / "object_geometry.jsonl", "w", encoding="utf-8")
            self._track_events_file = open(self._export_path / "tracking" / "track_events.jsonl", "w", encoding="utf-8")
            self._relation_events_file = open(self._export_path / "scene_graph" / "relation_events.jsonl", "w", encoding="utf-8")

            if not self.telemetry_log_path:
                self.telemetry_log_path = str(self._export_path / "slam" / "telemetry.csv")

            self.get_logger().info(f"Artifact exports initialized in {self._export_path}")

        if self.telemetry_log_path:
            p = Path(self.telemetry_log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._telemetry_file = open(p, "w", newline="", encoding="utf-8")
            self._telemetry_writer = csv.writer(self._telemetry_file)
            self._telemetry_writer.writerow([
                "timestamp", "frame_id", "pose_valid", "pose_age", "pose_source",
                "backend_name", "tracking_state", "matches", "inliers", "inlier_ratio",
                "features", "local_map_size", "local_key_frames", "key_frame_added",
                "registration_time_ms", "odometry_lost", "current_node_id",
                "loop_closure_id", "loop_closure_detected", "telemetry_age",
                "covariance_available", "latency_ms"
            ])
            trajectory_path = p.parent / "trajectory.tum"
            self._trajectory_file = open(trajectory_path, "w", encoding="utf-8")
            self.get_logger().info(f"Logging Stage 3 estimator telemetry to {self.telemetry_log_path} and trajectory to {trajectory_path}")

        self.imu_buffer: deque[IMUSample] = deque(maxlen=500)
        self.last_frame_timestamp: float | None = None
        if self.use_imu:
            self.imu_sub = self.create_subscription(
                Imu,
                self.imu_topic,
                self.imu_callback,
                100,
            )

        self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=4,
            slop=self.rgb_depth_max_dt,
        )
        self.sync.registerCallback(self.rgb_depth_callback)

        self.frame_counter = 0
        self.processed_frames = 0
        self.dropped_frames = 0
        self.processing_start_time: float | None = None
        self.last_input_time: float | None = None
        self.input_fps: float = 0.0

        max_q = max(0, self.queue_size)
        self.frame_queue = queue.Queue(maxsize=max_q)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(
            target=self._processing_worker,
            name="scene_graph_worker",
            daemon=True,
        )
        self.worker_thread.start()
        self.get_logger().info(
            f"SceneGraphROSNode initialized with worker thread.\n"
            f"  RGB: {self.rgb_topic}\n"
            f"  Depth: {self.depth_topic}\n"
            f"  CameraInfo: {self.camera_info_topic}\n"
            f"  TF: {self.world_frame} -> {self.sensor_frame} (max_dt={self.pose_max_dt}s)\n"
            f"  Queue: maxsize={max_q}, drop_old_frames={self.drop_old_frames}\n"
            f"  Object Cloud: {self.object_cloud_topic}\n"
            f"  Scene Cloud: {self.scene_cloud_topic} (enabled={self.publish_scene_cloud}, stride={self.scene_cloud_stride})\n"
            f"  Overlays: {self.overlay_detections_topic}, {self.overlay_tracks_topic}"
        )

    def camera_info_callback(self, msg: CameraInfo):
        """Cache latest valid camera info and convert to CameraIntrinsics."""
        self.latest_camera_info = msg
        try:
            self.latest_intrinsics = camera_info_to_intrinsics(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to parse CameraInfo: {e}")

    def imu_callback(self, msg: Imu):
        """Buffer incoming IMU samples into bounded ring buffer (500 samples, ~2.5s @ 200 Hz)."""
        sample = imu_msg_to_sample(msg)
        self.imu_buffer.append(sample)

    def _odom_info_callback(self, msg: Any):
        """Forward RTAB-Map /rtabmap/odom_info message to pose estimator."""
        if isinstance(self.pose_estimator, RTABMapEstimator):
            self.pose_estimator.add_odom_info_msg(msg)

    def _info_callback(self, msg: Any):
        """Forward RTAB-Map /rtabmap/info message to pose estimator."""
        if isinstance(self.pose_estimator, RTABMapEstimator):
            self.pose_estimator.add_info_msg(msg)

    def _odom_callback(self, msg: Any):
        """Forward RTAB-Map /rtabmap/odom message to pose estimator."""
        if isinstance(self.pose_estimator, RTABMapEstimator):
            self.pose_estimator.add_odom_msg(msg)

    def rgb_depth_callback(self, rgb_msg: Image, depth_msg: Image):
        """Lightweight callback: copies message references to queue and returns immediately."""
        now_wall = time.monotonic()
        if self.last_input_time is not None:
            dt = now_wall - self.last_input_time
            if dt > 0:
                inst_fps = 1.0 / dt
                self.input_fps = 0.9 * self.input_fps + 0.1 * inst_fps if self.input_fps > 0 else inst_fps
        self.last_input_time = now_wall

        self.metrics["input_rgbd_pairs"] += 1
        self.metrics["synchronized_frames"] += 1

        pending = PendingFrame(
            rgb_msg=rgb_msg,
            depth_msg=depth_msg,
            camera_info_msg=self.latest_camera_info,
            arrival_time=now_wall,
        )
        if self.drop_old_frames:
            try:
                self.frame_queue.put_nowait(pending)
            except queue.Full:
                try:
                    _ = self.frame_queue.get_nowait()
                    self.dropped_frames += 1
                    self.metrics["dropped_frames"] += 1
                    self.metrics["discard_reasons"]["queue_overflow"] += 1
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait(pending)
                except queue.Full:
                    self.dropped_frames += 1
                    self.metrics["dropped_frames"] += 1
                    self.metrics["discard_reasons"]["queue_overflow"] += 1
        else:
            try:
                self.frame_queue.put(pending, timeout=0.5)
            except queue.Full:
                self.dropped_frames += 1
                self.metrics["dropped_frames"] += 1
                self.metrics["discard_reasons"]["queue_overflow"] += 1
                self.get_logger().warning("Processing queue full; dropping frame")

    def _processing_worker(self):
        """Dedicated background worker executing FramePacket assembly and inference."""
        while not self.stop_event.is_set():
            try:
                pending = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._process_pending_frame(pending)
            except Exception as exc:
                import traceback
                self.get_logger().error(f"Error in processing worker: {exc}\n{traceback.format_exc()}", throttle_duration_sec=1.0)
            finally:
                self.frame_queue.task_done()

    def _process_pending_frame(self, pending: PendingFrame):
        t_frame_start = time.monotonic()
        if self.processing_start_time is None:
            self.processing_start_time = t_frame_start

        rgb_stamp = pending.rgb_msg.header.stamp
        rgb_timestamp = float(rgb_stamp.sec) + float(rgb_stamp.nanosec) * 1e-9
        intrinsics = None

        if pending.camera_info_msg is not None:
            try:
                intrinsics = camera_info_to_intrinsics(pending.camera_info_msg)
            except Exception:
                pass
        if intrinsics is None:
            intrinsics = self.latest_intrinsics
        if intrinsics is None:
            if self.config.camera is not None:
                intrinsics = CameraIntrinsics(
                    fx=self.config.camera.fx,
                    fy=self.config.camera.fy,
                    cx=self.config.camera.cx,
                    cy=self.config.camera.cy,
                    width=self.config.camera.width,
                    height=self.config.camera.height,
                )
            else:
                self.get_logger().warn("Waiting for CameraInfo before processing frames...")
                return

        depth_timestamp = rgb_timestamp
        if pending.depth_msg is not None:
            depth_stamp = pending.depth_msg.header.stamp
            depth_timestamp = float(depth_stamp.sec) + float(depth_stamp.nanosec) * 1e-9
            depth_skew = abs(rgb_timestamp - depth_timestamp)
            if depth_skew > self.rgb_depth_max_dt:
                self.get_logger().warn(
                    f"RGB-Depth desynchronization: skew={depth_skew*1000.0:.1f}ms > max={self.rgb_depth_max_dt*1000.0:.1f}ms",
                    throttle_duration_sec=2.0,
                )

        try:
            rgb_np = ros_image_to_numpy(pending.rgb_msg)
            depth_raw = ros_image_to_numpy(pending.depth_msg) if pending.depth_msg is not None else None
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return        

        session_id = "ros2_live_session"
        frame_seq = self.frame_counter
        runtime_depth_scale = None

        if runtime_depth_scale is not None and float(runtime_depth_scale) > 0:
            raw_scale = float(runtime_depth_scale)
            if raw_scale < 0.1:
                meters_per_unit = raw_scale
            else:
                meters_per_unit = 1.0 / raw_scale
        else:
            scale_param = self.depth_scale if self.depth_scale > 0 else 1000.0
            meters_per_unit = 1.0 / scale_param

        depth_m = None
        if depth_raw is not None:
            if np.issubdtype(depth_raw.dtype, np.integer):
                depth_m = depth_raw.astype(np.float32) * meters_per_unit
            else:
                depth_m = depth_raw.astype(np.float32)

        windowed_imu = ()
        if self.use_imu and self.last_frame_timestamp is not None:
            windowed_imu = tuple(
                s for s in self.imu_buffer
                if self.last_frame_timestamp < s.timestamp <= rgb_timestamp
            )

        self.last_frame_timestamp = rgb_timestamp
        sensor_status = StreamStatus.OK if depth_m is not None else StreamStatus.DEPTH_DROPPED        
        host_cap_stamp = None
        net_arr_stamp = None
        mapped_ts = rgb_timestamp

        frame_meta = {
            "source": "realsense2_camera",
            "source_timestamp": rgb_timestamp,
            "source_domain": "ros_system_time",
            "allow_camera_fallback": True,
        }
        if pending.depth_msg is not None:
            frame_meta["rgb_depth_dt_ms"] = float(depth_skew)

        sensor_frame = SensorFrame(
            session_id=session_id,
            sequence_number=frame_seq,
            timestamp=Timestamp(value=rgb_timestamp, domain=TimestampDomain.SYSTEM_TIME, source="ros_rgb"),
            rgb=rgb_np,
            camera_intrinsics=intrinsics,
            depth=depth_m,
            depth_scale=meters_per_unit,
            imu_samples=windowed_imu,
            host_capture_timestamp=host_cap_stamp,
            network_arrival_timestamp=net_arr_stamp,
            mapped_ros_timestamp=mapped_ts,
            frame_id=pending.rgb_msg.header.frame_id or self.sensor_frame,
            optical_frame_id=pending.depth_msg.header.frame_id if pending.depth_msg else self.sensor_frame,
            status=sensor_status,
            metadata=frame_meta,
        )

        t_est_start = time.monotonic()
        pose_estimate, estimator_state = self.pose_estimator.estimate(sensor_frame)
        estimate_latency_ms = (time.monotonic() - t_est_start) * 1000.0

        if self.localization_mode == "camera_local" or (self.config.reference_frame and self.config.reference_frame.type == "camera"):
            relation_frame = RelationReferenceFrame.create(
                "camera",
                pose_estimate.world_T_camera if pose_estimate.valid else np.eye(4, dtype=np.float64),
            )
        else:
            origin = pose_estimate.world_T_camera[:3, 3] if pose_estimate.valid else None
            relation_frame = self.ref_frame_provider.get_frame(origin_world=origin)

        world_frame_used = self.sensor_frame if self.localization_mode == "camera_local" else self.world_frame
        packet = FramePacket.from_sensor_frame(
            sensor_frame=sensor_frame,
            pose_estimate=pose_estimate,
            estimator_state=estimator_state,
            localization_mode=LocalizationMode.CAMERA_LOCAL_MODE if self.localization_mode == "camera_local" else LocalizationMode.WORLD_MODE,
            relation_frame=relation_frame,
            world_frame=world_frame_used,
        )

        if self.frame_counter % 25 == 0 or self.frame_counter < 10:
            cov_str = "YES" if estimator_state.covariance_available else "NO"
            loop_str = "YES" if estimator_state.loop_closure_detected else "NO"
            self.get_logger().info(
                f"[RTAB-MAP {estimator_state.tracking_state.value.upper()}] "
                f"frame={self.frame_counter:04d} | "
                f"pose_age={pose_estimate.age*1000.0:.1f}ms | "
                f"features={estimator_state.features} | "
                f"matches={estimator_state.matches} | "
                f"inliers={estimator_state.inliers} | "
                f"ratio={estimator_state.inlier_ratio:.3f} | "
                f"local_map={estimator_state.local_map_size} | "
                f"covariance={cov_str} | "
                f"loop_closure={loop_str} | "
                f"estimation={estimate_latency_ms:.1f}ms"
            )

        if self._telemetry_writer is not None:
            self._telemetry_writer.writerow([
                f"{rgb_timestamp:.6f}",
                self.frame_counter,
                pose_estimate.valid,
                f"{pose_estimate.age:.4f}" if np.isfinite(pose_estimate.age) else "-1.0",
                pose_estimate.source,
                estimator_state.backend_name,
                estimator_state.tracking_state.value,
                estimator_state.matches,
                estimator_state.inliers,
                f"{estimator_state.inlier_ratio:.4f}",
                estimator_state.features,
                estimator_state.local_map_size,
                estimator_state.local_key_frames,
                estimator_state.key_frame_added,
                f"{estimator_state.registration_time_ms:.2f}",
                estimator_state.odometry_lost,
                estimator_state.current_node_id,
                estimator_state.loop_closure_id,
                estimator_state.loop_closure_detected,
                f"{estimator_state.telemetry_age:.4f}" if np.isfinite(estimator_state.telemetry_age) else "-1.0",
                estimator_state.covariance_available,
                f"{estimate_latency_ms:.2f}",
            ])
            if self.frame_counter % 10 == 0 and self._telemetry_file is not None:
                self._telemetry_file.flush()

        if self._trajectory_file is not None and pose_estimate.valid and pose_estimate.world_T_camera is not None:
            T = pose_estimate.world_T_camera
            qx, qy, qz, qw = matrix_to_quaternion(T[:3, :3])
            tx, ty, tz = T[:3, 3]
            self._trajectory_file.write(
                f"{rgb_timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )
            if self.frame_counter % 10 == 0:
                self._trajectory_file.flush()

        if self.debug_frame_packet_only:
            self.get_logger().info(
                f"Frame {self.frame_counter} | "
                f"RGB: {rgb_np.shape} | "
                f"Depth: {depth_m.shape if depth_m is not None else 'None'} | "
                f"Timestamp: {rgb_timestamp:.4f} | "
                f"Pose: {'available' if pose_estimate.valid else 'missing'}"
            )
            self.frame_counter += 1
            self.processed_frames += 1
            return

        t_infer_start = time.monotonic()
        try:
            max_pose_age = getattr(self.pose_estimator, "slam_pose_max_age", 0.20)
            if hasattr(self, "config") and hasattr(self.config, "estimation") and hasattr(self.config.estimation, "max_pose_age"):
                max_pose_age = self.config.estimation.max_pose_age
            elif hasattr(self, "config") and hasattr(self.config, "pose") and hasattr(self.config.pose, "max_pose_age"):
                max_pose_age = self.config.pose.max_pose_age

            has_pose = bool(pose_estimate.valid and packet.world_T_camera is not None and getattr(packet, "transform_valid", False) is True)
            is_stale_source = getattr(packet, "transform_source", "") in ("stale_tf", "stale_slam_tf")
            age = getattr(packet, "pose_age", float("inf"))
            is_stale_age = np.isfinite(age) and (age > max_pose_age)
            is_stale = is_stale_source or (has_pose and is_stale_age)
            is_global_valid = bool(
                has_pose
                and getattr(packet, "is_persistent_world_frame", True) is True
                and not is_stale
            )
            if is_global_valid:
                self.metrics["frames_with_valid_slam_pose"] += 1
            else:
                self.metrics["frames_with_unlocalized_or_missing_pose"] += 1
                self.metrics["discard_reasons"]["unlocalized_pose"] += 1

            graph = self.pipeline.update(packet)
            infer_latency_ms = (time.monotonic() - t_infer_start) * 1000.0
            t_now = time.monotonic()
            total_latency_ms = (t_now - pending.arrival_time) * 1000.0
            elapsed = t_now - self.processing_start_time
            proc_fps = (self.processed_frames + 1) / elapsed if elapsed > 0 else 0.0

            self.metrics["graph_updates"] += 1
            self.metrics["latencies_ms"]["estimation"].append(estimate_latency_ms)
            self.metrics["latencies_ms"]["inference"].append(infer_latency_ms)
            self.metrics["latencies_ms"]["total"].append(total_latency_ms)

            snapshot = create_snapshot(
                graph=graph,
                packet=packet,
                input_fps=self.input_fps,
                processing_fps=proc_fps,
                total_latency_ms=total_latency_ms,
                queue_size=self.frame_queue.qsize(),
                dropped_frames=self.dropped_frames,
                tf_latency_ms=estimate_latency_ms,
                inference_latency_ms=infer_latency_ms,
            )

            observations = getattr(self.pipeline, "last_observations", None)
            new_observations = getattr(self.pipeline, "new_observations", None)
            if new_observations is None and not hasattr(self.pipeline, "new_observations"):
                new_observations = observations

            self.graph_publisher.publish(
                graph=graph,
                packet=packet,
                snapshot=snapshot,
                observations=observations,
            )

            if new_observations is not None:
                self.metrics["detector_invocations"] += 1
                for obs in new_observations:
                    self.metrics["total_detections"] += 1
                    bbox_coords = [float(c) for c in obs.bbox_xyxy] if getattr(obs, "bbox_xyxy", None) is not None else ([float(c) for c in obs.bbox] if getattr(obs, "bbox", None) is not None else None)
                    obs_f_idx = getattr(obs, "frame_index", self.frame_counter)
                    obs_t_stamp = getattr(obs, "timestamp", rgb_timestamp)
                    if self._detections_file is not None:
                        det_rec = {
                            "frame_index": obs_f_idx,
                            "timestamp": obs_t_stamp,
                            "obs_id": getattr(obs, "obs_id", ""),
                            "class_name": getattr(obs, "class_name", ""),
                            "confidence": float(getattr(obs, "confidence", 0.0)),
                            "bbox": bbox_coords,
                            "inference_time_ms": round(infer_latency_ms, 2),
                        }
                        self._detections_file.write(json.dumps(det_rec) + "\n")

                    geom = getattr(obs, "object_geometry", None)
                    if geom is not None:
                        is_localized = (getattr(geom, "localization_status", "unlocalized").lower() == "localized")
                        is_depth_ok = bool(
                            geom.status in (GeometryStatus.VALID, GeometryStatus.NO_POSE, GeometryStatus.STALE_POSE)
                            and geom.centroid_camera is not None
                        )

                        if is_localized and is_depth_ok:
                            self.metrics["primary_geometry_outcomes"]["localized_valid_geometry"] += 1
                        elif is_localized and not is_depth_ok:
                            self.metrics["primary_geometry_outcomes"]["localized_insufficient_depth"] += 1
                            self.metrics["discard_reasons"]["no_depth_or_points"] += 1
                        elif not is_localized and is_depth_ok:
                            self.metrics["primary_geometry_outcomes"]["unlocalized_valid_geometry"] += 1
                        else:
                            self.metrics["primary_geometry_outcomes"]["unlocalized_insufficient_depth"] += 1
                            self.metrics["discard_reasons"]["no_depth_or_points"] += 1

                        self.metrics["valid_3d_geometries"] = self.metrics["primary_geometry_outcomes"]["localized_valid_geometry"]
                        self.metrics["unlocalized_geometries"] = (
                            self.metrics["primary_geometry_outcomes"]["unlocalized_valid_geometry"]
                            + self.metrics["primary_geometry_outcomes"]["unlocalized_insufficient_depth"]
                        )
                        self.metrics["invalid_or_out_of_range_depth_geometries"] = (
                            self.metrics["primary_geometry_outcomes"]["localized_insufficient_depth"]
                            + self.metrics["primary_geometry_outcomes"]["unlocalized_insufficient_depth"]
                        )

                        if self._geometry_file is not None:
                            coord_frame = getattr(geom, "coordinate_frame", "camera")
                            loc_status = getattr(geom, "localization_status", "unlocalized").lower()
                            source_val = getattr(geom, "source", "slam" if is_localized else "local_fallback")
                            pose_ts_val = getattr(geom, "pose_timestamp", None)
                            geom_status_str = geom.status.name if hasattr(geom.status, "name") else str(geom.status)

                            geom_rec = {
                                "frame_index": obs_f_idx,
                                "timestamp": obs_t_stamp,
                                "obs_id": getattr(obs, "obs_id", ""),
                                "class_name": getattr(obs, "class_name", ""),
                                "coordinate_frame": coord_frame,
                                "localization_status": loc_status,
                                "pose_timestamp": pose_ts_val,
                                "status": geom_status_str,
                                "centroid_camera": [round(float(v), 4) for v in geom.centroid_camera] if geom.centroid_camera is not None else None,
                                "centroid_world": [round(float(v), 4) for v in geom.centroid_world] if (is_localized and geom.centroid_world is not None) else None,
                                "bbox_min_world": [round(float(v), 4) for v in geom.bbox_min_world] if (is_localized and geom.bbox_min_world is not None) else None,
                                "bbox_max_world": [round(float(v), 4) for v in geom.bbox_max_world] if (is_localized and geom.bbox_max_world is not None) else None,
                                "valid_point_count": int(geom.valid_point_count) if geom.valid_point_count is not None else 0,
                                "source": source_val,
                            }
                            self._geometry_file.write(json.dumps(geom_rec) + "\n")

            # Check tracker updates
            tracker = None
            if hasattr(self.pipeline, "core") and hasattr(self.pipeline.core, "tracker"):
                tracker = self.pipeline.core.tracker
            elif hasattr(self.pipeline, "_async_impl") and hasattr(self.pipeline._async_impl.core, "tracker"):
                tracker = self.pipeline._async_impl.core.tracker
            if tracker is not None:
                self.metrics["tracker_updates"] += 1

            # Save snapshot and visualization periodically
            if self._export_path is not None:
                if self.frame_counter % 25 == 0 or self.frame_counter < 5:
                    snap_path = self._export_path / "scene_graph" / "snapshots" / f"snapshot_{self.frame_counter:05d}.json"
                    with open(snap_path, "w", encoding="utf-8") as sf:
                        json.dump(snapshot.to_dict(), sf, indent=2)

                # Render overlay frame
                overlay_frame = render_tracks_overlay(rgb_np, graph, frame_index=self.frame_counter)

                # Video writer
                if self._video_writer is None:
                    h, w = overlay_frame.shape[:2]
                    video_path = str(self._export_path / "perception" / "overlay.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self._video_writer = cv2.VideoWriter(video_path, fourcc, 15.0, (w, h))
                if self._video_writer is not None:
                    bgr_overlay = cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR)
                    self._video_writer.write(bgr_overlay)

                # Save gallery frames (early, middle, late)
                if self.frame_counter in (10, 50, 100, 250, 500, 1000, 1500, 2000) or (self.frame_counter % 500 == 0):
                    vis_path = self._export_path / "visualizations" / f"overlay_frame_{self.frame_counter:05d}.png"
                    cv2.imwrite(str(vis_path), cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR))
                    self._gallery_saved += 1
        except Exception as e:
            self.get_logger().error(f"Error executing scene graph pipeline on frame {self.frame_counter}: {e}")

        self.frame_counter += 1
        self.processed_frames += 1

        if self.processed_frames % 20 == 0:
            elapsed = time.monotonic() - self.processing_start_time
            rate = self.processed_frames / elapsed if elapsed > 0 else 0.0
            telem = self.pipeline.get_telemetry() if hasattr(self.pipeline, "get_telemetry") else {}
            det_rate = telem.get("detection_rate_hz", 0.0)
            det_lat = telem.get("last_detector_latency_ms", 0.0)
            self.get_logger().info(
                f"Pipeline stats: tracking={rate:.1f} Hz, detection={det_rate:.1f} Hz (lat={det_lat:.1f}ms), "
                f"queue_size={self.frame_queue.qsize()}, dropped={self.dropped_frames}"
            )

    def destroy_node(self):
        """Clean shutdown stopping worker thread, exporting artifacts and summary."""
        self.stop_event.set()
        if hasattr(self, "pipeline") and self.pipeline is not None and hasattr(self.pipeline, "stop"):
            self.pipeline.stop()
        if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)

        if hasattr(self, "_video_writer") and self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        if hasattr(self, "_detections_file") and self._detections_file is not None and not self._detections_file.closed:
            self._detections_file.flush()
            self._detections_file.close()

        if hasattr(self, "_geometry_file") and self._geometry_file is not None and not self._geometry_file.closed:
            self._geometry_file.flush()
            self._geometry_file.close()

        if hasattr(self, "_telemetry_file") and self._telemetry_file is not None and not self._telemetry_file.closed:
            self._telemetry_file.flush()
            self._telemetry_file.close()

        if hasattr(self, "_trajectory_file") and self._trajectory_file is not None and not self._trajectory_file.closed:
            self._trajectory_file.flush()
            self._trajectory_file.close()

        if getattr(self, "_export_path", None) is not None and hasattr(self, "pipeline") and self.pipeline is not None:
            try:
                tracker = None
                if hasattr(self.pipeline, "core") and hasattr(self.pipeline.core, "tracker"):
                    tracker = self.pipeline.core.tracker
                elif hasattr(self.pipeline, "_async_impl") and hasattr(self.pipeline._async_impl.core, "tracker"):
                    tracker = self.pipeline._async_impl.core.tracker

                if tracker is not None:
                    tracks_csv_path = self._export_path / "tracking" / "tracks.csv"
                    with open(tracks_csv_path, "w", newline="", encoding="utf-8") as tf:
                        writer = csv.writer(tf)
                        writer.writerow([
                            "track_id", "class_name", "state", "observation_count", "missing_count",
                            "first_frame", "last_frame", "centroid_x", "centroid_y", "centroid_z"
                        ])
                        for t_id, track in tracker.tracks.items():
                            c_world = track.centroid_world
                            cx = round(float(c_world[0]), 4) if c_world is not None else ""
                            cy = round(float(c_world[1]), 4) if c_world is not None else ""
                            cz = round(float(c_world[2]), 4) if c_world is not None else ""
                            state_val = track.state.name if hasattr(track.state, "name") else str(track.state)
                            obs_count = getattr(track, "observation_count", getattr(track, "hits", 0))
                            first_f = getattr(track, "first_observed_frame", getattr(track, "first_frame_index", -1))
                            last_f = getattr(track, "last_observed_frame", getattr(track, "last_frame_index", -1))
                            writer.writerow([
                                t_id, track.class_name, state_val, obs_count,
                                track.missing_count, first_f, last_f, cx, cy, cz
                            ])
                            if track.state == TrackState.ACTIVE:
                                self.metrics["confirmed_tracks"] += 1

                core_pipeline = getattr(self.pipeline, "core", None) or getattr(getattr(self.pipeline, "_async_impl", None), "core", None)
                if core_pipeline is not None and hasattr(core_pipeline, "graph"):
                    graph = core_pipeline.graph
                    self.metrics["active_nodes_final"] = len(graph.get_active_nodes())
                    self.metrics["active_edges_final"] = len(graph.get_active_edges())
                    self.metrics["emitted_relations"] = len(graph.edges)

                    sg_json_path = self._export_path / "scene_graph" / "scene_graph.json"
                    with open(sg_json_path, "w", encoding="utf-8") as sgf:
                        nodes_dict = {}
                        for k, node in graph.nodes.items():
                            tr = getattr(node, "track", None)
                            c_w = tr.centroid_world if tr is not None and hasattr(tr, "centroid_world") else None
                            nodes_dict[str(k)] = {
                                "track_id": str(k),
                                "class_name": getattr(node, "class_name", getattr(tr, "class_name", "unknown")),
                                "state": node.state.name if hasattr(node, "state") and hasattr(node.state, "name") else str(getattr(node, "state", "UNKNOWN")),
                                "is_active": getattr(node, "is_active", True),
                                "centroid_world": [round(float(v), 4) for v in c_w] if c_w is not None else None,
                            }
                        edges_list = []
                        for k, edge in graph.edges.items():
                            sub = getattr(edge, "subject_id", k[0] if isinstance(k, tuple) and len(k) > 0 else "")
                            pred = getattr(edge, "canonical_predicate", getattr(edge, "predicate", k[2] if isinstance(k, tuple) and len(k) > 2 else ""))
                            obj = getattr(edge, "object_id", k[1] if isinstance(k, tuple) and len(k) > 1 else "")
                            st = edge.state.name if hasattr(edge, "state") and hasattr(edge.state, "name") else str(getattr(edge, "state", "ACTIVE"))
                            cf = float(edge.latest_evidence.confidence) if hasattr(edge, "latest_evidence") and edge.latest_evidence is not None else float(getattr(edge, "confidence", 1.0))
                            edges_list.append({
                                "subject": sub,
                                "predicate": pred,
                                "object": obj,
                                "state": st,
                                "confidence": cf,
                            })
                        json.dump({
                            "nodes": nodes_dict,
                            "edges": edges_list,
                        }, sgf, indent=2)

                    # Also export entities.csv
                    entities_csv_path = self._export_path / "scene_graph" / "entities.csv"
                    with open(entities_csv_path, "w", newline="", encoding="utf-8") as ef:
                        ewriter = csv.writer(ef)
                        ewriter.writerow(["entity_id", "class_name", "state", "is_active", "x", "y", "z"])
                        for k, n_info in nodes_dict.items():
                            cw = n_info.get("centroid_world") or ["", "", ""]
                            ewriter.writerow([k, n_info["class_name"], n_info["state"], n_info["is_active"], cw[0], cw[1], cw[2]])

                    if hasattr(graph, "history") and hasattr(graph.history, "events"):
                        if self._relation_events_file is not None:
                            for ev in graph.history.events:
                                ev_rec = {
                                    "event_type": ev.event_type.name if hasattr(ev.event_type, "name") else str(ev.event_type),
                                    "timestamp": float(ev.timestamp),
                                    "frame_index": int(ev.frame_index),
                                    "subject_id": str(ev.subject_id),
                                    "object_id": str(ev.object_id),
                                    "predicate": str(ev.predicate),
                                }
                                self._relation_events_file.write(json.dumps(ev_rec) + "\n")

                if self._relation_events_file is not None and not self._relation_events_file.closed:
                    self._relation_events_file.flush()
                    self._relation_events_file.close()

                if self._track_events_file is not None and not self._track_events_file.closed:
                    self._track_events_file.flush()
                    self._track_events_file.close()

                lats = self.metrics.pop("latencies_ms", {})
                metrics_summary = dict(self.metrics)
                for k_lat, v_list in lats.items():
                    if v_list:
                        arr = np.array(v_list, dtype=np.float64)
                        metrics_summary[f"latency_{k_lat}_p50_ms"] = round(float(np.percentile(arr, 50)), 2)
                        metrics_summary[f"latency_{k_lat}_p95_ms"] = round(float(np.percentile(arr, 95)), 2)
                        metrics_summary[f"latency_{k_lat}_mean_ms"] = round(float(np.mean(arr)), 2)

                with open(self._export_path / "metrics.json", "w", encoding="utf-8") as mf:
                    json.dump(metrics_summary, mf, indent=2)

                with open(self._export_path / "summary.json", "w", encoding="utf-8") as sf:
                    json.dump({
                        "status": "COMPLETED",
                        "total_processed_frames": self.processed_frames,
                        "metrics": metrics_summary,
                    }, sf, indent=2)
                self.get_logger().info(f"Artifact exports saved successfully to {self._export_path}")
            except Exception as exp:
                self.get_logger().error(f"Failed to finalize artifact exports: {exp}")

        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SceneGraphROSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()