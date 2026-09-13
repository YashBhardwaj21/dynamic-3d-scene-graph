"""TUM ROS Player Node.

The ONLY component allowed to know about TUM dataset specifics.
Publishes standard ROS 2 messages:
- /tum/rgb/image_raw (sensor_msgs/msg/Image)
- /tum/depth/image_raw (sensor_msgs/msg/Image)
- /tum/rgb/camera_info (sensor_msgs/msg/CameraInfo)
- /tf (geometry_msgs/msg/TransformStamped: world -> camera_optical_frame)
"""

import time
from pathlib import Path
from typing import Optional, List, Dict
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros

from scene_graph.data.tum_loader import TUMLoader, RGBEntry, DepthEntry, PoseEntry
from scene_graph.data.synchronization import associate
from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph_ros.config_loader import load_scene_graph_config, resolve_path
from scene_graph_ros.ros_conversions import (
    numpy_to_ros_image,
    intrinsics_to_camera_info,
    matrix_to_transform_stamped,
)


class TUMPlayerNode(Node):
    """ROS 2 node that replays TUM RGB-D sequences with standard ROS messages."""

    def __init__(self):
        super().__init__("tum_player")

        # Declare parameters
        self.declare_parameter("dataset_root", "data/raw/rgbd_dataset_freiburg1_desk")
        self.declare_parameter("config_path", "configs/tum_fr1_desk.yaml")
        self.declare_parameter("rate_multiplier", 1.0)
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("end_frame", -1)
        self.declare_parameter("rgb_topic", "/tum/rgb/image_raw")
        self.declare_parameter("depth_topic", "/tum/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/tum/rgb/camera_info")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("sensor_frame", "camera_optical_frame")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("loop", False)

        # Retrieve parameters
        dataset_root_param = self.get_parameter("dataset_root").get_parameter_value().string_value
        config_path_param = self.get_parameter("config_path").get_parameter_value().string_value
        self.rate_multiplier = self.get_parameter("rate_multiplier").get_parameter_value().double_value
        self.param_start_frame = self.get_parameter("start_frame").get_parameter_value().integer_value
        self.param_end_frame = self.get_parameter("end_frame").get_parameter_value().integer_value
        self.rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.world_frame = self.get_parameter("world_frame").get_parameter_value().string_value
        self.sensor_frame = self.get_parameter("sensor_frame").get_parameter_value().string_value
        self.publish_rate_hz = self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        self.loop = self.get_parameter("loop").get_parameter_value().bool_value

        # Load configuration
        self.config = load_scene_graph_config(config_path_param)

        # Resolve dataset root
        dataset_root = resolve_path(dataset_root_param)
        if not dataset_root.exists() and self.config.dataset is not None:
            dataset_root = resolve_path(self.config.dataset.root)

        if not dataset_root.exists():
            raise FileNotFoundError(f"TUM dataset root not found at {dataset_root}")

        self.get_logger().info(f"Loading TUM sequence from {dataset_root}")
        self.loader = TUMLoader(dataset_root)
        self.rgb_entries: List[RGBEntry] = self.loader.load_rgb()
        self.depth_entries: List[DepthEntry] = self.loader.load_depth()
        self.pose_entries: List[PoseEntry] = self.loader.load_groundtruth()

        # Synchronize indices
        rgb_timestamps = [e.timestamp for e in self.rgb_entries]
        depth_timestamps = [e.timestamp for e in self.depth_entries]
        pose_timestamps = [e.timestamp for e in self.pose_entries]

        rgb_depth_max_dt = self.config.sync.rgb_depth_max_dt if self.config.sync else 0.02
        rgb_pose_max_dt = self.config.sync.rgb_pose_max_dt if self.config.sync else 0.02

        self.rgb_to_depth: Dict[int, int] = dict(associate(rgb_timestamps, depth_timestamps, rgb_depth_max_dt))
        self.rgb_to_pose: Dict[int, int] = dict(associate(rgb_timestamps, pose_timestamps, rgb_pose_max_dt))

        # Frame boundaries
        start_frame = self.param_start_frame
        end_frame = self.param_end_frame

        if start_frame == 0 and self.config.sequence is not None:
            start_frame = self.config.sequence.start_frame
        if end_frame == -1 and self.config.sequence is not None and self.config.sequence.end_frame is not None:
            end_frame = self.config.sequence.end_frame
        if end_frame == -1:
            end_frame = len(self.rgb_entries) - 1

        self.start_frame = max(0, min(start_frame, len(self.rgb_entries) - 1))
        self.end_frame = min(end_frame, len(self.rgb_entries) - 1)
        self.current_idx = self.start_frame

        # Camera intrinsics
        if self.config.camera is not None:
            self.intrinsics = CameraIntrinsics(
                fx=self.config.camera.fx,
                fy=self.config.camera.fy,
                cx=self.config.camera.cx,
                cy=self.config.camera.cy,
                width=self.config.camera.width,
                height=self.config.camera.height,
            )
        else:
            # Standard Freiburg 1 defaults if missing
            self.intrinsics = CameraIntrinsics(
                fx=525.0, fy=525.0, cx=319.5, cy=239.5, width=640, height=480
            )

        # Publishers
        self.rgb_pub = self.create_publisher(Image, self.rgb_topic, 10)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, self.camera_info_topic, 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Preload frames and pre-build all ROS messages before starting timer
        self.preload_frames()

        # Playback metrics
        self.published_count = 0
        self.start_wall_time: Optional[float] = None

        # Playback timer
        timer_period = 1.0 / max(1.0, self.publish_rate_hz)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        total_frames = self.end_frame - self.start_frame + 1
        self.get_logger().info(
            f"TUM Player initialized: frames {self.start_frame}..{self.end_frame} ({total_frames} total), "
            f"target_rate={self.publish_rate_hz:.1f}Hz"
        )

    def preload_frames(self):
        """Preload selected TUM frames and construct all ROS messages in RAM."""
        total_frames = self.end_frame - self.start_frame + 1
        self.get_logger().info(
            f"Preloading {total_frames} frames ({self.start_frame}..{self.end_frame}) into RAM..."
        )
        t0 = time.monotonic()

        self.rgb_msgs: List[Image] = []
        self.depth_msgs: List[Optional[Image]] = []
        self.camera_info_msgs: List[CameraInfo] = []
        self.tf_msgs: List[Optional[object]] = []

        for idx in range(self.start_frame, self.end_frame + 1):
            rgb_entry = self.rgb_entries[idx]
            current_ts = rgb_entry.timestamp

            # 1. RGB
            rgb_path = str(self.loader.resolve_rgb_path(rgb_entry))
            rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                raise RuntimeError(f"Failed to read RGB: {rgb_path}")

            rgb_arr = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            self.rgb_msgs.append(
                numpy_to_ros_image(
                    rgb_arr,
                    encoding="rgb8",
                    frame_id=self.sensor_frame,
                    timestamp=current_ts,
                )
            )

            # 2. Depth
            depth_msg = None
            if idx in self.rgb_to_depth:
                d_idx = self.rgb_to_depth[idx]
                depth_path = str(self.loader.resolve_depth_path(self.depth_entries[d_idx]))
                depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                if depth_raw is None:
                    raise RuntimeError(f"Failed to read depth: {depth_path}")

                depth_msg = numpy_to_ros_image(
                    depth_raw,
                    encoding="16UC1",
                    frame_id=self.sensor_frame,
                    timestamp=self.depth_entries[d_idx].timestamp,
                )
            self.depth_msgs.append(depth_msg)

            # 3. CameraInfo
            self.camera_info_msgs.append(
                intrinsics_to_camera_info(
                    self.intrinsics,
                    frame_id=self.sensor_frame,
                    timestamp=current_ts,
                )
            )

            # 4. TF
            tf_msg = None
            if idx in self.rgb_to_pose:
                p_idx = self.rgb_to_pose[idx]
                pose_matrix = self.pose_entries[p_idx].as_transform_matrix()
                tf_msg = matrix_to_transform_stamped(
                    pose_matrix,
                    parent_frame=self.world_frame,
                    child_frame=self.sensor_frame,
                    timestamp=self.pose_entries[p_idx].timestamp,
                )
            self.tf_msgs.append(tf_msg)

        duration = time.monotonic() - t0
        self.get_logger().info(
            f"Preloaded {len(self.rgb_msgs)} frames in {duration:.2f}s. Ready for playback."
        )

    def timer_callback(self):
        if self.current_idx > self.end_frame:
            if self.loop:
                self.get_logger().info("Sequence finished. Looping back to start.")
                self.current_idx = self.start_frame
                self.start_wall_time = time.monotonic()
                self.published_count = 0
            else:
                elapsed = time.monotonic() - (self.start_wall_time or time.monotonic())
                actual_rate = self.published_count / max(elapsed, 1e-9)
                self.get_logger().info(
                    f"Sequence playback finished: {self.published_count} frames | "
                    f"target_rate={self.publish_rate_hz:.1f} Hz, actual_rate={actual_rate:.1f} Hz"
                )
                self.timer.cancel()
                return

        if self.start_wall_time is None:
            self.start_wall_time = time.monotonic()

        offset = self.current_idx - self.start_frame

        # Publish pre-built ROS messages
        self.rgb_pub.publish(self.rgb_msgs[offset])

        if self.depth_msgs[offset] is not None:
            self.depth_pub.publish(self.depth_msgs[offset])

        self.camera_info_pub.publish(self.camera_info_msgs[offset])

        if self.tf_msgs[offset] is not None:
            self.tf_broadcaster.sendTransform(self.tf_msgs[offset])

        self.published_count += 1

        if self.current_idx % 50 == 0 or self.current_idx == self.start_frame:
            elapsed = time.monotonic() - self.start_wall_time
            actual_rate = self.published_count / max(elapsed, 1e-9)
            self.get_logger().info(
                f"Published frame {self.current_idx}/{self.end_frame} | "
                f"target_rate={self.publish_rate_hz:.1f} Hz, actual_rate={actual_rate:.1f} Hz"
            )

        self.current_idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = TUMPlayerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
