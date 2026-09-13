"""Generic SceneGraph ROS Node.

Knows only generic concepts:
- RGB image (sensor_msgs/msg/Image)
- Depth image (sensor_msgs/msg/Image)
- Camera calibration (sensor_msgs/msg/CameraInfo)
- Camera pose via TF (geometry_msgs/msg/TransformStamped: world -> camera_optical_frame)
- Optional IMU samples (sensor_msgs/msg/Imu)

Contains NO dataset-specific (TUM, RealSense, D455, etc.) logic.
Separates ROS message ingestion from scene-graph inference using a dedicated
worker thread and frame queue. OnlinePipeline never executes in the ROS callback.
"""

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

from scene_graph.config import SceneGraphConfig
from scene_graph.data.frame_packet import FramePacket, IMUSample
from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.geometry.reference_frame import RelationReferenceFrame
from scene_graph.pipeline.online_pipeline import OnlinePipeline

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


class SceneGraphROSNode(Node):
    """Generic ROS 2 node orchestrating online 3D scene graph generation with asynchronous worker."""

    def __init__(self):
        super().__init__("scene_graph_node")

        # Declare parameters (all topics, frames, and queue parameters configurable)
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
        self.declare_parameter("depth_scale", 5000.0)
        self.declare_parameter("debug_frame_packet_only", False)
        self.declare_parameter("state_topic", "/scene_graph/state")
        self.declare_parameter("markers_topic", "/scene_graph/markers")
        self.declare_parameter("queue_size", 64)
        self.declare_parameter("drop_old_frames", False)

        # Read parameters
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
        self.queue_size = self.get_parameter("queue_size").get_parameter_value().integer_value
        self.drop_old_frames = self.get_parameter("drop_old_frames").get_parameter_value().bool_value

        self.get_logger().info(f"Loading SceneGraph configuration from {self.config_path}")
        self.config: SceneGraphConfig = load_scene_graph_config(self.config_path)

        # Determine depth model
        depth_scale = self.param_depth_scale
        if self.config.depth is not None and self.config.depth.scale > 0:
            depth_scale = self.config.depth.scale
        self.depth_model = DepthModel(scale=depth_scale)

        # Initialize online pipeline (unless in debug isolation mode)
        if not self.debug_frame_packet_only:
            self.get_logger().info("Initializing OnlinePipeline...")
            self.pipeline = OnlinePipeline(self.config)
            self.get_logger().info("OnlinePipeline ready.")
        else:
            self.pipeline = None
            self.get_logger().warn("Running in debug_frame_packet_only mode: Core pipeline bypassed.")

        # Publisher helper
        self.graph_publisher = GraphPublisher(
            node=self,
            state_topic=self.state_topic,
            markers_topic=self.markers_topic,
            world_frame=self.world_frame,
        )

        # CameraInfo caching
        self.latest_camera_info: Optional[CameraInfo] = None
        self.latest_intrinsics: Optional[CameraIntrinsics] = None
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        # TF2 listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Optional IMU buffer
        self.imu_buffer: List[IMUSample] = []
        self.last_frame_timestamp: Optional[float] = None
        if self.use_imu:
            self.imu_sub = self.create_subscription(
                Imu,
                self.imu_topic,
                self.imu_callback,
                100,
            )

        # Synchronized RGB and Depth subscribers
        self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)

        # ApproximateTimeSynchronizer with configurable slop
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=30,
            slop=self.rgb_depth_max_dt,
        )
        self.sync.registerCallback(self.rgb_depth_callback)

        # Frame counter & performance statistics
        self.frame_counter = 0
        self.processed_frames = 0
        self.processing_start_time: Optional[float] = None

        # Dedicated worker thread and bounded frame queue
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
            f"  Queue: maxsize={max_q}, drop_old_frames={self.drop_old_frames}"
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
        """Lightweight callback: copies message references to queue and returns immediately.
        
        TF callbacks and other ROS callbacks remain completely unblocked.
        """
        pending = PendingFrame(
            rgb_msg=rgb_msg,
            depth_msg=depth_msg,
            camera_info_msg=self.latest_camera_info,
        )

        if self.drop_old_frames:
            try:
                self.frame_queue.put_nowait(pending)
            except queue.Full:
                self.get_logger().warning("Processing queue full; dropping incoming frame")
        else:
            try:
                self.frame_queue.put(pending, timeout=0.5)
            except queue.Full:
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
        """Heavy processing: TF lookup, image conversion, FramePacket creation, and OnlinePipeline."""
        if self.processing_start_time is None:
            self.processing_start_time = time.monotonic()

        rgb_stamp = pending.rgb_msg.header.stamp
        rgb_timestamp = float(rgb_stamp.sec) + float(rgb_stamp.nanosec) * 1e-9

        # 1. Resolve intrinsics
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

        # 2. Lookup camera pose via TF (TF listener remains unblocked on executor thread!)
        world_T_camera = None
        target_time = Time(seconds=rgb_stamp.sec, nanoseconds=rgb_stamp.nanosec)
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.sensor_frame,
                target_time,
                timeout=Duration(seconds=self.pose_max_dt),
            )
            world_T_camera = transform_to_matrix(transform_stamped)
        except TransformException as ex:
            self.get_logger().warn(
                f"TF lookup failed for {self.world_frame} -> {self.sensor_frame} at {rgb_timestamp:.4f}: {ex}"
            )

        # 3. Convert images to numpy arrays
        try:
            rgb_np = ros_image_to_numpy(pending.rgb_msg)
            depth_np = ros_image_to_numpy(pending.depth_msg)
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        # 4. Windowed IMU samples in (last_ts, current_ts]
        windowed_imu = ()
        if self.use_imu and self.last_frame_timestamp is not None:
            windowed_imu = tuple(
                s for s in self.imu_buffer
                if self.last_frame_timestamp < s.timestamp <= rgb_timestamp
            )
        self.last_frame_timestamp = rgb_timestamp

        # 5. Construct generic relation reference frame
        up_axis = np.array([0.0, 0.0, 1.0])
        heading_axis = np.array([1.0, 0.0, 0.0])
        if self.config.reference_frame is not None:
            up_axis = np.array(self.config.reference_frame.up_axis, dtype=np.float64)
            heading_axis = np.array(self.config.reference_frame.heading_axis, dtype=np.float64)

        relation_frame = RelationReferenceFrame.from_gravity_and_heading(
            origin_world=np.zeros(3),
            up_axis_world=up_axis,
            heading_world=heading_axis,
        )

        # 6. Construct dataset-independent FramePacket
        packet = FramePacket(
            frame_index=self.frame_counter,
            timestamp=rgb_timestamp,
            rgb=rgb_np,
            depth=depth_np,
            world_T_camera=world_T_camera,
            camera_intrinsics=intrinsics,
            depth_model=self.depth_model,
            relation_frame=relation_frame,
            imu_samples=windowed_imu,
            metadata={"frame_id": pending.rgb_msg.header.frame_id},
        )

        # Acceptance Test 2 Mode: Verify FramePacket construction without running core
        if self.debug_frame_packet_only:
            self.get_logger().info(
                f"Frame {self.frame_counter} | "
                f"RGB: {rgb_np.shape} | "
                f"Depth: {depth_np.shape if depth_np is not None else 'None'} | "
                f"Timestamp: {rgb_timestamp:.4f} | "
                f"Pose: {'available' if world_T_camera is not None else 'missing'} | "
                f"Intrinsics: available | "
                f"IMU samples: {len(windowed_imu)}"
            )
            self.frame_counter += 1
            self.processed_frames += 1
            return

        # 7. Feed FramePacket into OnlinePipeline
        try:
            graph = self.pipeline.update(packet)
            self.graph_publisher.publish(graph, packet)
        except Exception as e:
            self.get_logger().error(f"Error executing scene graph pipeline on frame {self.frame_counter}: {e}")

        self.frame_counter += 1
        self.processed_frames += 1

        # Periodic statistics logging
        if self.processed_frames % 20 == 0:
            elapsed = time.monotonic() - self.processing_start_time
            rate = self.processed_frames / elapsed if elapsed > 0 else 0.0
            self.get_logger().info(
                f"Inference stats: processed={self.processed_frames}, "
                f"rate={rate:.2f} Hz, queue_size={self.frame_queue.qsize()}"
            )

    def destroy_node(self):
        """Clean shutdown stopping worker thread before node destruction."""
        self.stop_event.set()
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
