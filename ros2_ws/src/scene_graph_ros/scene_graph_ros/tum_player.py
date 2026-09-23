"""TUM RGB-D sequence player node."""

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

import time
from typing import Optional, List, Dict
import cv2
import numpy as np

import rcl_interfaces.msg
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
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
    """Replays TUM RGB-D sequences as ROS messages."""

    def __init__(self):
        super().__init__("tum_player")

        self.declare_parameter("dataset_root", "data/raw/rgbd_dataset_freiburg1_desk")
        self.declare_parameter("config_path", "configs/tum_fr1_desk.yaml")
        self.declare_parameter("rate_multiplier", 1.0)
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("end_frame", -1)
        self.declare_parameter("frame_stride", 1)
        self.declare_parameter("rgb_topic", "/tum/rgb/image_raw")
        self.declare_parameter("depth_topic", "/tum/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/tum/rgb/camera_info")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("sensor_frame", "camera_optical_frame")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("publish_groundtruth_tf",
            False,
            rcl_interfaces.msg.ParameterDescriptor(
                description="Publish ground-truth world→sensor TF transforms. "
                            "Set to false when a SLAM backend (e.g. RTAB-Map) "
                            "provides its own TF instead."
            ),
        )
        # Override depth scale (raw integer units per metre).
        # If 0 (default), falls back to config.depth.scale or the dataset default.
        self.declare_parameter("depth_scale", 0.0)
        # Float32 (metres) depth topic published alongside the raw 16UC1 topic.
        # RTAB-Map consumes this directly so it never needs to know the depth scale.
        # Set to empty string to disable.
        self.declare_parameter("slam_depth_topic", "/tum/depth_m/image_raw")

        dataset_root_param = self.get_parameter("dataset_root").get_parameter_value().string_value
        config_path_param = self.get_parameter("config_path").get_parameter_value().string_value
        self.rate_multiplier = self.get_parameter("rate_multiplier").get_parameter_value().double_value
        self.param_start_frame = self.get_parameter("start_frame").get_parameter_value().integer_value
        self.param_end_frame = self.get_parameter("end_frame").get_parameter_value().integer_value
        self.frame_stride = max(1, self.get_parameter("frame_stride").get_parameter_value().integer_value)
        self.rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.world_frame = self.get_parameter("world_frame").get_parameter_value().string_value
        self.sensor_frame = self.get_parameter("sensor_frame").get_parameter_value().string_value
        self.publish_rate_hz = self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        self.loop = self.get_parameter("loop").get_parameter_value().bool_value
        self.publish_groundtruth_tf = (
            self.get_parameter("publish_groundtruth_tf").get_parameter_value().bool_value
        )
        _depth_scale_param = self.get_parameter("depth_scale").get_parameter_value().double_value
        self.slam_depth_topic = self.get_parameter("slam_depth_topic").get_parameter_value().string_value.strip()

        self.config = load_scene_graph_config(config_path_param)

        # Resolve depth scale: parameter > config > dataset default
        if _depth_scale_param > 0:
            self.depth_scale = float(_depth_scale_param)
        elif self.config.depth is not None and self.config.depth.scale > 0:
            self.depth_scale = float(self.config.depth.scale)
        else:
            self.depth_scale = 5000.0  # TUM standard default

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

        rgb_timestamps = [e.timestamp for e in self.rgb_entries]
        depth_timestamps = [e.timestamp for e in self.depth_entries]
        pose_timestamps = [e.timestamp for e in self.pose_entries]

        rgb_depth_max_dt = self.config.sync.rgb_depth_max_dt if self.config.sync else 0.02
        rgb_pose_max_dt = self.config.sync.rgb_pose_max_dt if self.config.sync else 0.02

        self.rgb_to_depth: Dict[int, int] = dict(associate(rgb_timestamps, depth_timestamps, rgb_depth_max_dt))
        self.rgb_to_pose: Dict[int, int] = dict(associate(rgb_timestamps, pose_timestamps, rgb_pose_max_dt))

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
            self.intrinsics = CameraIntrinsics(
                fx=525.0, fy=525.0, cx=319.5, cy=239.5, width=640, height=480
            )

        self.rgb_pub = self.create_publisher(Image, self.rgb_topic, 10)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, self.camera_info_topic, 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # /clock drives simulated time for RTAB-Map, scene_graph_node, and RViz.
        # tum_player itself does NOT use use_sim_time — its timer must run on wall clock.
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)


        # Float32 (metres) depth topic for RTAB-Map — avoids all depth-scale confusion.
        self.slam_depth_pub = None
        if self.slam_depth_topic:
            self.slam_depth_pub = self.create_publisher(Image, self.slam_depth_topic, 10)
            self.get_logger().info(
                f"Publishing float32 depth (metres) for SLAM on {self.slam_depth_topic} "
                f"(scale={self.depth_scale:.0f} raw units/m)"
            )

        self.frame_cache: Dict[int, Tuple[Image, Optional[Image], CameraInfo, Optional[object]]] = {}

        self.published_count = 0
        self.start_wall_time: Optional[float] = None

        timer_period = 1.0 / max(1.0, self.publish_rate_hz)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        total_frames = self.end_frame - self.start_frame + 1
        effective_frames = (total_frames + self.frame_stride - 1) // self.frame_stride
        tf_mode = "groundtruth TF" if self.publish_groundtruth_tf else "no TF (SLAM backend expected)"
        self.get_logger().info(
            f"TUM Player ready: frames {self.start_frame}..{self.end_frame} "
            f"({total_frames} total, stride={self.frame_stride} → {effective_frames} to publish), "
            f"target_rate={self.publish_rate_hz:.1f}Hz, pose_mode={tf_mode}"
        )

    def get_frame_messages(self, idx: int):
        """Fetch or lazily load and convert frame messages on demand."""
        if idx in self.frame_cache:
            return self.frame_cache[idx]

        rgb_entry = self.rgb_entries[idx]
        current_ts = rgb_entry.timestamp

        rgb_path = str(self.loader.resolve_rgb_path(rgb_entry))
        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"Failed to read RGB: {rgb_path}")

        rgb_arr = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb_msg = numpy_to_ros_image(
            rgb_arr,
            encoding="rgb8",
            frame_id=self.sensor_frame,
            timestamp=current_ts,
        )

        depth_msg = None
        slam_depth_msg = None
        if idx in self.rgb_to_depth:
            d_idx = self.rgb_to_depth[idx]
            depth_path = str(self.loader.resolve_depth_path(self.depth_entries[d_idx]))
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth_raw is not None:
                depth_msg = numpy_to_ros_image(
                    depth_raw,
                    encoding="16UC1",
                    frame_id=self.sensor_frame,
                    timestamp=current_ts,
                )
                # Float32 metres depth for SLAM backends (RTAB-Map, etc.)
                if self.slam_depth_pub is not None:
                    depth_m = (depth_raw.astype(np.float32) / self.depth_scale)
                    slam_depth_msg = numpy_to_ros_image(
                        depth_m,
                        encoding="32FC1",
                        frame_id=self.sensor_frame,
                        timestamp=current_ts,
                    )

        camera_info_msg = intrinsics_to_camera_info(
            self.intrinsics,
            frame_id=self.sensor_frame,
            timestamp=current_ts,
        )

        tf_msg = None
        if idx in self.rgb_to_pose:
            p_idx = self.rgb_to_pose[idx]
            pose_matrix = self.pose_entries[p_idx].as_transform_matrix()
            tf_msg = matrix_to_transform_stamped(
                pose_matrix,
                parent_frame=self.world_frame,
                child_frame=self.sensor_frame,
                timestamp=current_ts,
            )

        cached = (rgb_msg, depth_msg, slam_depth_msg, camera_info_msg, tf_msg)
        if len(self.frame_cache) < 500:
            self.frame_cache[idx] = cached
        return cached

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

        if self.published_count == 0 and self.rgb_pub.get_subscription_count() == 0:
            self.get_logger().info(
                "Waiting for downstream consumers (scene_graph_node) to subscribe before starting playback...",
                throttle_duration_sec=3.0,
            )
            return

        if self.start_wall_time is None:
            self.start_wall_time = time.monotonic()

        rgb_msg, depth_msg, slam_depth_msg, camera_info_msg, tf_msg = self.get_frame_messages(self.current_idx)

        # Advance simulated time FIRST so downstream nodes (RTAB-Map, scene_graph_node,
        # RViz) see the correct TUM timestamp before the frame data arrives.
        clock_msg = Clock()
        clock_msg.clock = rgb_msg.header.stamp
        self.clock_pub.publish(clock_msg)

        if tf_msg is not None and self.publish_groundtruth_tf:
            self.tf_broadcaster.sendTransform(tf_msg)

        self.camera_info_pub.publish(camera_info_msg)


        if depth_msg is not None:
            self.depth_pub.publish(depth_msg)

        # Publish float32 metres depth for SLAM backends (RTAB-Map)
        if slam_depth_msg is not None and self.slam_depth_pub is not None:
            self.slam_depth_pub.publish(slam_depth_msg)

        self.rgb_pub.publish(rgb_msg)

        self.published_count += 1

        if self.current_idx % 50 == 0 or self.current_idx == self.start_frame:
            elapsed = time.monotonic() - self.start_wall_time
            actual_rate = self.published_count / max(elapsed, 1e-9)
            self.get_logger().info(
                f"Published frame {self.current_idx}/{self.end_frame} | "
                f"target_rate={self.publish_rate_hz:.1f} Hz, actual_rate={actual_rate:.1f} Hz"
            )

        self.current_idx += self.frame_stride



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
