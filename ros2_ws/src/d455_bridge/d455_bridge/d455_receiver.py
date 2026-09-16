import json
import socket
import struct

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo


HOST = "0.0.0.0"
PORT = 5000


def recv_exact(conn, size):
    data = bytearray()

    while len(data) < size:
        chunk = conn.recv(min(1024 * 1024, size - len(data)))

        if not chunk:
            raise ConnectionError("Sender disconnected")

        data.extend(chunk)

    return bytes(data)


class D455Receiver(Node):

    def __init__(self):
        super().__init__("d455_receiver")

        self.rgb_pub = self.create_publisher(
            Image,
            "/camera/camera/color/image_raw",
            2,
        )

        self.depth_pub = self.create_publisher(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            2,
        )

        self.camera_info_pub = self.create_publisher(
            CameraInfo,
            "/camera/camera/color/camera_info",
            2,
        )

        self.get_logger().info(
            f"Starting D455 TCP receiver on {HOST}:{PORT}"
        )

        self.server = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        self.server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        self.server.bind((HOST, PORT))
        self.server.listen(1)

        self.server.settimeout(1.0)

        self.conn = None

        self.timer = self.create_timer(
            0.001,
            self.process_connection,
        )

        self.waiting_logged = False

    def process_connection(self):
        if self.conn is None:

            try:
                conn, addr = self.server.accept()

                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)

                self.conn = conn
                self.conn.settimeout(None)

                self.get_logger().info(
                    f"D455 sender connected from {addr}"
                )

            except socket.timeout:
                if not self.waiting_logged:
                    self.get_logger().info(
                        f"Waiting for Windows sender on "
                        f"{HOST}:{PORT}"
                    )
                    self.waiting_logged = True

                return

        try:
            self.receive_frame()

        except ConnectionError as exc:
            self.get_logger().warn(str(exc))
            self.conn.close()
            self.conn = None

        except Exception as exc:
            self.get_logger().error(
                f"Frame processing error: {type(exc).__name__}: {exc}"
            )

            self.conn.close()
            self.conn = None

    def receive_frame(self):

        # ------------------------------------------------------------
        # Header length
        # ------------------------------------------------------------

        raw_size = recv_exact(
            self.conn,
            4,
        )

        header_size = struct.unpack(
            "!I",
            raw_size,
        )[0]

        if not 1 <= header_size <= 1024 * 1024:
            raise ValueError(
                f"Invalid header size: {header_size}"
            )

        # ------------------------------------------------------------
        # Header
        # ------------------------------------------------------------

        header_bytes = recv_exact(
            self.conn,
            header_size,
        )

        header = json.loads(
            header_bytes.decode("utf-8")
        )

        if header.get("type") != "d455_rgbd":
            raise ValueError(
                f"Unexpected packet type: "
                f"{header.get('type')}"
            )

        frame_id = header["frame_id"]

        color = header["color"]
        depth = header["depth"]

        rgb_width = color["width"]
        rgb_height = color["height"]

        depth_width = depth["width"]
        depth_height = depth["height"]

        rgb_size = color["payload_size"]
        depth_size = depth["payload_size"]

        # ------------------------------------------------------------
        # Payloads
        # ------------------------------------------------------------

        rgb_bytes = recv_exact(
            self.conn,
            rgb_size,
        )

        depth_bytes = recv_exact(
            self.conn,
            depth_size,
        )

        # ------------------------------------------------------------
        # Validate
        # ------------------------------------------------------------

        expected_rgb = (
            rgb_width
            * rgb_height
            * color["channels"]
            * color["bytes_per_channel"]
        )

        expected_depth = (
            depth_width
            * depth_height
            * depth["channels"]
            * depth["bytes_per_channel"]
        )

        if len(rgb_bytes) != expected_rgb:
            raise ValueError(
                f"RGB size mismatch: "
                f"{len(rgb_bytes)} != {expected_rgb}"
            )

        if len(depth_bytes) != expected_depth:
            raise ValueError(
                f"Depth size mismatch: "
                f"{len(depth_bytes)} != {expected_depth}"
            )

        # ------------------------------------------------------------
        # ROS timestamp
        #
        # Use node's current clock so timestamps align precisely with the
        # local ROS 2 TF tree and eliminate host-WSL clock drift.
        # ------------------------------------------------------------
        now_stamp = self.get_clock().now().to_msg()

        # ------------------------------------------------------------
        # RGB message
        # ------------------------------------------------------------

        rgb_msg = Image()

        rgb_msg.header.stamp = now_stamp
        rgb_msg.header.frame_id = "camera_color_optical_frame"

        rgb_msg.height = rgb_height
        rgb_msg.width = rgb_width
        rgb_msg.encoding = "bgr8"
        rgb_msg.is_bigendian = 0
        rgb_msg.step = rgb_width * 3
        rgb_msg.data.frombytes(rgb_bytes)

        # ------------------------------------------------------------
        # Depth message
        # ------------------------------------------------------------

        depth_msg = Image()

        depth_msg.header.stamp = now_stamp
        depth_msg.header.frame_id = "camera_color_optical_frame"

        depth_msg.height = depth_height
        depth_msg.width = depth_width
        depth_msg.encoding = "16UC1"
        depth_msg.is_bigendian = 0
        depth_msg.step = depth_width * 2
        depth_msg.data.frombytes(depth_bytes)

        # ------------------------------------------------------------
        # CameraInfo
        # ------------------------------------------------------------

        camera_info = CameraInfo()

        camera_info.header.stamp = now_stamp
        camera_info.header.frame_id = "camera_color_optical_frame"

        camera_info.width = rgb_width
        camera_info.height = rgb_height

        # Intrinsics are filled by the Windows sender.
        camera_info.k = [
            color["fx"],
            0.0,
            color["ppx"],
            0.0,
            color["fy"],
            color["ppy"],
            0.0,
            0.0,
            1.0,
        ]

        camera_info.d = color.get(
            "distortion",
            [],
        )

        camera_info.distortion_model = color.get(
            "distortion_model",
            "",
        )

        # Identity rectification matrix
        camera_info.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]

        # 3x4 projection matrix [K | 0]
        camera_info.p = [
            color["fx"], 0.0, color["ppx"], 0.0,
            0.0, color["fy"], color["ppy"], 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        # ------------------------------------------------------------
        # Publish
        # ------------------------------------------------------------

        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.camera_info_pub.publish(camera_info)

        if frame_id % 30 == 0:
            self.get_logger().info(
                f"Frame {frame_id} | "
                f"RGB {rgb_width}x{rgb_height} | "
                f"Depth {depth_width}x{depth_height} | "
                f"dt={header['rgb_depth_dt_ms']:.2f} ms"
            )

    def destroy_node(self):

        if self.conn is not None:
            self.conn.close()

        self.server.close()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = D455Receiver()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
