"""Generic SceneGraph ROS Node."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repository root and active virtualenvs are on sys.path
def _ensure_paths():
    current = Path(__file__).resolve().parent
    while current != current.parent:
        candidate_src = current / "src"
        if (candidate_src / "scene_graph").exists():
            src_str = str(candidate_src)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            break
        current = current.parent

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        for py_ver in ["python3.10", "python3.11", "python3.9", "python3"]:
            sp = Path(venv) / "lib" / py_ver / "site-packages"
            if sp.exists() and str(sp) not in sys.path:
                sys.path.insert(0, str(sp))
    home_myenv = Path.home() / "myenv" / "lib" / "python3.10" / "site-packages"
    if home_myenv.exists() and str(home_myenv) not in sys.path:
        sys.path.insert(0, str(home_myenv))

_ensure_paths()

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo, Imu
import message_filters
import tf2_ros
from tf2_ros import TransformException

from scene_graph.data.frame_packet import FramePacket, IMUSample, LocalizationMode
from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.geometry.reference_frame import RelationReferenceFrame
from scene_graph.pipeline.online_pipeline import OnlinePipeline
from scene_graph.graph.snapshot import create_snapshot

from scene_graph_ros.config_loader import load_scene_graph_config
from scene_graph_ros.ros_conversions import (
    ros_image_to_numpy,
    camera_info_to_intrinsics,
    transform_to_matrix,
    imu_msg_to_sample,
)
from scene_graph_ros.graph_publisher import GraphPublisher


@dataclass
class PendingFrame:
    """Container for synchronized raw ROS messages awaiting worker processing."""
    rgb_msg: Image
    depth_msg: Image
    camera_info_msg: Optional[CameraInfo]
    arrival_time: float


class SceneGraphROSNode(Node):
    """ROS 2 node orchestrating online 3D scene graph generation."""

    def __init__(self):
        super().__init__("scene_graph_node")

        self.declare_parameter("rgb_topic", "/tum/rgb/image_raw")
        self.declare_parameter("depth_topic", "/tum/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/tum/rgb/camera_info")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("sensor_frame", "camera_optical_frame")
        self.declare_parameter("rgb_depth_max_dt", 0.02)
        self.declare_parameter("pose_max_dt", 0.05)
        self.declare_parameter("use_imu", False)
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("config_path", "configs/default.yaml")
        self.declare_parameter("depth_scale", 1000.0)
        self.declare_parameter("debug_frame_packet_only", False)
        self.declare_parameter("state_topic", "/scene_graph/state")
        self.declare_parameter("markers_topic", "/scene_graph/markers")
        self.declare_parameter("object_cloud_topic", "/scene_graph/object_cloud")
        self.declare_parameter("scene_cloud_topic", "/scene_graph/scene_cloud")
        self.declare_parameter("publish_scene_cloud", True)
        self.declare_parameter("scene_cloud_stride", 4)
        self.declare_parameter("overlay_detections_topic", "/scene_graph/overlay_detections")
        self.declare_parameter("overlay_tracks_topic", "/scene_graph/overlay_tracks")
        self.declare_parameter("queue_size", 64)
        self.declare_parameter("drop_old_frames", False)
        self.declare_parameter("localization_mode", "world")
        self.declare_parameter("async_mode", True)
        self.declare_parameter("min_hits", -1)
        self.declare_parameter("max_missing_seconds", -1.0)

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
        self.overlay_detections_topic = self.get_parameter("overlay_detections_topic").get_parameter_value().string_value
        self.overlay_tracks_topic = self.get_parameter("overlay_tracks_topic").get_parameter_value().string_value
        self.queue_size = self.get_parameter("queue_size").get_parameter_value().integer_value
        self.drop_old_frames = self.get_parameter("drop_old_frames").get_parameter_value().bool_value
        self.localization_mode = self.get_parameter("localization_mode").get_parameter_value().string_value.lower()
        self.async_mode = self.get_parameter("async_mode").get_parameter_value().bool_value
        self.param_min_hits = self.get_parameter("min_hits").get_parameter_value().integer_value
        self.param_max_missing_seconds = self.get_parameter("max_missing_seconds").get_parameter_value().double_value


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

        if not self.debug_frame_packet_only:
            self.get_logger().info(f"Initializing OnlinePipeline (async_mode={self.async_mode})...")
            self.pipeline = OnlinePipeline(self.config, async_mode=self.async_mode)
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
            overlay_detections_topic=self.overlay_detections_topic,
            overlay_tracks_topic=self.overlay_tracks_topic,
            world_frame=self.world_frame,
        )

        self.latest_camera_info: Optional[CameraInfo] = None
        self.latest_intrinsics: Optional[CameraIntrinsics] = None
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.imu_buffer: List[IMUSample] = []
        self.last_frame_timestamp: Optional[float] = None
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
            queue_size=30,
            slop=self.rgb_depth_max_dt,
        )
        self.sync.registerCallback(self.rgb_depth_callback)

        self.frame_counter = 0
        self.processed_frames = 0
        self.dropped_frames = 0
        self.processing_start_time: Optional[float] = None
        self.last_input_time: Optional[float] = None
        self.input_fps: float = 0.0

        max_q = self.queue_size if self.queue_size > 0 else 0
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
        """Buffer incoming IMU samples."""
        sample = imu_msg_to_sample(msg)
        self.imu_buffer.append(sample)
        if len(self.imu_buffer) > 500:
            self.imu_buffer.pop(0)

    def rgb_depth_callback(self, rgb_msg: Image, depth_msg: Image):
        """Lightweight callback: copies message references to queue and returns immediately."""
        now_wall = time.monotonic()
        if self.last_input_time is not None:
            dt = now_wall - self.last_input_time
            if dt > 0:
                inst_fps = 1.0 / dt
                self.input_fps = 0.9 * self.input_fps + 0.1 * inst_fps if self.input_fps > 0 else inst_fps
        self.last_input_time = now_wall

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
                    # Drop oldest frame to ensure fresh state
                    _ = self.frame_queue.get_nowait()
                    self.dropped_frames += 1
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait(pending)
                except queue.Full:
                    self.dropped_frames += 1
        else:
            try:
                self.frame_queue.put(pending, timeout=0.5)
            except queue.Full:
                self.dropped_frames += 1
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
                self.get_logger().error(f"Error in processing worker: {exc}", throttle_duration_sec=1.0)
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

        # Check depth synchronization
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

        world_T_camera = None
        pose_timestamp = None
        pose_age = float("inf")
        transform_valid = False
        transform_source = "unknown"
        tf_latency_ms = 0.0

        if self.localization_mode == "camera_local":
            world_T_camera = np.eye(4, dtype=np.float64)
            pose_timestamp = rgb_timestamp
            pose_age = 0.0
            transform_valid = True
            transform_source = "camera_local"
            world_frame_used = self.sensor_frame
        else:
            # WORLD_MODE requires valid, timestamp-correct localization
            world_frame_used = self.world_frame
            t_tf_start = time.monotonic()
            target_time = Time(seconds=rgb_stamp.sec, nanoseconds=rgb_stamp.nanosec)
            try:
                transform_stamped = self.tf_buffer.lookup_transform(
                    self.world_frame,
                    self.sensor_frame,
                    target_time,
                    timeout=Duration(seconds=self.pose_max_dt),
                )
                tf_stamp = transform_stamped.header.stamp
                pose_timestamp = float(tf_stamp.sec) + float(tf_stamp.nanosec) * 1e-9
                pose_age = abs(rgb_timestamp - pose_timestamp)

                if pose_age > self.pose_max_dt:
                    transform_valid = False
                    transform_source = "stale_tf"
                    self.get_logger().warn(
                        f"Stale TF for {self.world_frame} -> {self.sensor_frame} at {rgb_timestamp:.4f}: age={pose_age*1000.0:.1f}ms > max={self.pose_max_dt*1000.0:.1f}ms",
                        throttle_duration_sec=2.0,
                    )
                else:
                    world_T_camera = transform_to_matrix(transform_stamped)
                    transform_valid = True
                    transform_source = "tf_exact"
            except TransformException as ex:
                transform_valid = False
                transform_source = "missing_tf"
                self.get_logger().warn(
                    f"TF lookup failed for {self.world_frame} -> {self.sensor_frame} at {rgb_timestamp:.4f}: {ex}. Frame marked invalid for world graph.",
                    throttle_duration_sec=2.0,
                )
            tf_latency_ms = (time.monotonic() - t_tf_start) * 1000.0

        try:
            rgb_np = ros_image_to_numpy(pending.rgb_msg)
            depth_raw = ros_image_to_numpy(pending.depth_msg) if pending.depth_msg is not None else None
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        depth_m = None
        if depth_raw is not None:
            if np.issubdtype(depth_raw.dtype, np.integer):
                scale = self.depth_scale if self.depth_scale > 0 else 1000.0
                depth_m = depth_raw.astype(np.float32) / scale
            else:
                depth_m = depth_raw.astype(np.float32)

        windowed_imu = ()
        if self.use_imu and self.last_frame_timestamp is not None:
            windowed_imu = tuple(
                s for s in self.imu_buffer
                if self.last_frame_timestamp < s.timestamp <= rgb_timestamp
            )
        self.last_frame_timestamp = rgb_timestamp

        if self.localization_mode == "camera_local" or (self.config.reference_frame and self.config.reference_frame.type == "camera"):
            relation_frame = RelationReferenceFrame.create("camera", world_T_camera if world_T_camera is not None else np.eye(4, dtype=np.float64))
        else:
            up_axis = np.array(self.config.reference_frame.up_axis, dtype=np.float64) if self.config.reference_frame else np.array([0.0, 0.0, 1.0])
            heading_axis = np.array(self.config.reference_frame.heading_axis, dtype=np.float64) if self.config.reference_frame else np.array([1.0, 0.0, 0.0])
            relation_frame = RelationReferenceFrame.from_gravity_and_heading(
                origin_world=world_T_camera[:3, 3] if world_T_camera is not None else np.zeros(3, dtype=np.float64),
                up_axis_world=up_axis,
                heading_world=heading_axis,
            )

        packet = FramePacket(
            frame_index=self.frame_counter,
            timestamp=rgb_timestamp,
            rgb=rgb_np,
            depth=depth_m,
            world_T_camera=world_T_camera,
            camera_intrinsics=intrinsics,
            depth_model=self.depth_model,
            relation_frame=relation_frame,
            imu_samples=windowed_imu,
            frame_id=pending.rgb_msg.header.frame_id or self.sensor_frame,
            optical_frame_id=pending.depth_msg.header.frame_id if pending.depth_msg else self.sensor_frame,
            pose_source=transform_source,
            sensor_timestamp=rgb_timestamp,
            rgb_timestamp=rgb_timestamp,
            depth_timestamp=depth_timestamp,
            pose_timestamp=pose_timestamp,
            pose_age=pose_age if np.isfinite(pose_age) else 0.0,
            world_frame=world_frame_used,
            transform_source=transform_source,
            transform_valid=transform_valid,
            localization_mode=LocalizationMode.CAMERA_LOCAL_MODE if self.localization_mode == "camera_local" else LocalizationMode.WORLD_MODE,
            metadata={"frame_id": pending.rgb_msg.header.frame_id},
        )


        if self.debug_frame_packet_only:
            self.get_logger().info(
                f"Frame {self.frame_counter} | "
                f"RGB: {rgb_np.shape} | "
                f"Depth: {depth_np.shape if depth_np is not None else 'None'} | "
                f"Timestamp: {rgb_timestamp:.4f} | "
                f"Pose: {'available' if world_T_camera is not None else 'missing'}"
            )
            self.frame_counter += 1
            self.processed_frames += 1
            return

        t_infer_start = time.monotonic()
        try:
            graph = self.pipeline.update(packet)
            infer_latency_ms = (time.monotonic() - t_infer_start) * 1000.0

            t_now = time.monotonic()
            total_latency_ms = (t_now - pending.arrival_time) * 1000.0
            elapsed = t_now - self.processing_start_time
            proc_fps = (self.processed_frames + 1) / elapsed if elapsed > 0 else 0.0

            snapshot = create_snapshot(
                graph=graph,
                packet=packet,
                input_fps=self.input_fps,
                processing_fps=proc_fps,
                total_latency_ms=total_latency_ms,
                queue_size=self.frame_queue.qsize(),
                dropped_frames=self.dropped_frames,
                tf_latency_ms=tf_latency_ms,
                inference_latency_ms=infer_latency_ms,
            )

            self.graph_publisher.publish(
                graph=graph,
                packet=packet,
                snapshot=snapshot,
                observations=getattr(self.pipeline, "last_observations", None),
            )
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
        """Clean shutdown stopping worker thread before node destruction."""
        self.stop_event.set()
        if hasattr(self, "pipeline") and self.pipeline is not None and hasattr(self.pipeline, "stop"):
            self.pipeline.stop()
        if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
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
