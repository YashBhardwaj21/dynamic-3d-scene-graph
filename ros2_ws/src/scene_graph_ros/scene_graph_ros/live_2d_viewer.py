"""Live 2D perception viewer node for detections, tracks, and telemetry."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

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

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from scene_graph_ros.ros_conversions import ros_image_to_numpy


class Live2DViewerNode(Node):
    """ROS 2 viewer displaying detection and tracking overlays side-by-side with telemetry."""

    def __init__(self):
        super().__init__("live_2d_viewer")

        self.declare_parameter("detections_topic", "/scene_graph/overlay_detections")
        self.declare_parameter("tracks_topic", "/scene_graph/overlay_tracks")
        self.declare_parameter("state_topic", "/scene_graph/state")
        self.declare_parameter("window_name", "Dynamic 3D Scene Graph - Live 2D Perception")
        self.declare_parameter("display_rate_hz", 30.0)

        self.detections_topic = self.get_parameter("detections_topic").get_parameter_value().string_value
        self.tracks_topic = self.get_parameter("tracks_topic").get_parameter_value().string_value
        self.state_topic = self.get_parameter("state_topic").get_parameter_value().string_value
        self.window_name = self.get_parameter("window_name").get_parameter_value().string_value
        display_rate = self.get_parameter("display_rate_hz").get_parameter_value().double_value

        self.latest_det_img: Optional[np.ndarray] = None
        self.latest_track_img: Optional[np.ndarray] = None
        self.latest_telemetry: dict = {}
        self.latest_summary: dict = {}
        self.frame_index: int = 0
        self.gui_available: bool = True

        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            self.get_logger().warn(
                "No DISPLAY or WAYLAND_DISPLAY found in environment. GUI window disabled (headless mode)."
            )
            self.gui_available = False

        self.det_sub = self.create_subscription(
            Image,
            self.detections_topic,
            self.detections_callback,
            10,
        )
        self.track_sub = self.create_subscription(
            Image,
            self.tracks_topic,
            self.tracks_callback,
            10,
        )
        self.state_sub = self.create_subscription(
            String,
            self.state_topic,
            self.state_callback,
            10,
        )

        period = 1.0 / max(1.0, display_rate)
        self.render_timer = self.create_timer(period, self.render_callback)

        self.get_logger().info(
            f"Live2DViewerNode initialized.\n"
            f"  Detections: {self.detections_topic}\n"
            f"  Tracks: {self.tracks_topic}\n"
            f"  State: {self.state_topic}"
        )

    def detections_callback(self, msg: Image):
        try:
            self.latest_det_img = ros_image_to_numpy(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to decode detection image: {e}")

    def tracks_callback(self, msg: Image):
        try:
            self.latest_track_img = ros_image_to_numpy(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to decode track image: {e}")

    def state_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.latest_telemetry = data.get("telemetry", {})
            self.latest_summary = data.get("summary", {})
            self.frame_index = data.get("frame", 0)
        except Exception as e:
            self.get_logger().warn(f"Failed to parse state JSON: {e}")

    def _draw_header(self, canvas: np.ndarray, title: str, x: int, y: int, width: int):
        cv2.rectangle(canvas, (x, y), (x + width, y + 28), (17, 23, 31), -1)
        cv2.putText(
            canvas,
            title,
            (x + 10, y + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 220, 230),
            1,
            cv2.LINE_AA,
        )

    def render_callback(self):
        if not self.gui_available:
            return

        if self.latest_det_img is None and self.latest_track_img is None:
            return

        ref_img = self.latest_track_img if self.latest_track_img is not None else self.latest_det_img
        h, w = ref_img.shape[:2]

        det_panel = self.latest_det_img if self.latest_det_img is not None else np.zeros((h, w, 3), dtype=np.uint8)
        track_panel = self.latest_track_img if self.latest_track_img is not None else np.zeros((h, w, 3), dtype=np.uint8)

        det_bgr = cv2.cvtColor(det_panel, cv2.COLOR_RGB2BGR) if det_panel.ndim == 3 else cv2.cvtColor(det_panel, cv2.COLOR_GRAY2BGR)
        track_bgr = cv2.cvtColor(track_panel, cv2.COLOR_RGB2BGR) if track_panel.ndim == 3 else cv2.cvtColor(track_panel, cv2.COLOR_GRAY2BGR)

        combined_h = h + 32 + 42
        combined_w = w * 2 + 10
        dashboard = np.full((combined_h, combined_w, 3), 9, dtype=np.uint8)

        self._draw_header(dashboard, "CURRENT DETECTIONS & MASKS", 0, 0, w)
        self._draw_header(dashboard, "TRACKED OBJECTS & 3D SCENE GRAPH RELATIONS", w + 10, 0, w)

        dashboard[32:32 + h, 0:w] = det_bgr
        dashboard[32:32 + h, w + 10:w * 2 + 10] = track_bgr

        bar_y = 32 + h
        cv2.rectangle(dashboard, (0, bar_y), (combined_w, combined_h), (16, 22, 30), -1)
        cv2.line(dashboard, (0, bar_y), (combined_w, bar_y), (35, 45, 58), 1)

        fps = self.latest_telemetry.get("processing_fps", 0.0)
        in_fps = self.latest_telemetry.get("input_fps", 0.0)
        lat = self.latest_telemetry.get("total_latency_ms", 0.0)
        infer_lat = self.latest_telemetry.get("inference_latency_ms", 0.0)
        objs = self.latest_summary.get("active_objects", 0)
        rels = self.latest_summary.get("active_relations", 0)
        q = self.latest_telemetry.get("queue_size", 0)
        drops = self.latest_telemetry.get("dropped_frames", 0)

        status_text = (
            f"FRAME: {self.frame_index:04d}  |  "
            f"PROC: {fps:.1f} FPS  |  "
            f"INPUT: {in_fps:.1f} FPS  |  "
            f"LATENCY: {lat:.1f}ms (Infer: {infer_lat:.1f}ms)  |  "
            f"OBJECTS: {objs}  |  "
            f"RELATIONS: {rels}  |  "
            f"QUEUE: {q}  |  "
            f"DROPPED: {drops}"
        )

        cv2.putText(
            dashboard,
            status_text,
            (14, bar_y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (110, 225, 180),
            1,
            cv2.LINE_AA,
        )

        try:
            cv2.imshow(self.window_name, dashboard)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                self.get_logger().info("Exit key pressed, shutting down viewer...")
                rclpy.shutdown()
        except Exception as e:
            self.get_logger().warn(f"OpenCV GUI display error: {e}", throttle_duration_sec=5.0)
            self.gui_available = False

    def destroy_node(self):
        if self.gui_available:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Live2DViewerNode()
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
