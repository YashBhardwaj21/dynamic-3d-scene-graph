import json
import queue
import socket
import struct
import threading
import time
from typing import Optional, Dict, Any

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from sensor_msgs.msg import Image, CameraInfo, Imu
    from builtin_interfaces.msg import Time as TimeMsg
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False

    class Header:
        def __init__(self, stamp=None, frame_id=""):
            self.stamp = stamp
            self.frame_id = frame_id

    class TimeMsg:
        def __init__(self, sec=0, nanosec=0):
            self.sec = sec
            self.nanosec = nanosec

    class Image:
        def __init__(self):
            self.header = Header()
            self.height = 0
            self.width = 0
            self.encoding = ""
            self.step = 0
            self.data = bytearray()

    class CameraInfo:
        def __init__(self):
            self.header = Header()
            self.width = 0
            self.height = 0
            self.k = [0.0] * 9
            self.d = []
            self.distortion_model = ""
            self.r = [0.0] * 9
            self.p = [0.0] * 12

    class Vector3:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class Imu:
        def __init__(self):
            self.header = Header()
            self.linear_acceleration = Vector3()
            self.angular_velocity = Vector3()

    class Node:
        def __init__(self, name: str):
            self.name = name
            self._params = {}
            self._logger = self._MockLogger()

        class _MockLogger:
            def info(self, msg): pass
            def warn(self, msg): pass
            def error(self, msg): pass

        def declare_parameter(self, name, default):
            self._params[name] = default

        def get_parameter(self, name):
            class _Param:
                def __init__(self, val):
                    self._val = val
                def get_parameter_value(self):
                    return self
                @property
                def string_value(self):
                    return str(self._val)
                @property
                def integer_value(self):
                    return int(self._val)
                @property
                def double_value(self):
                    return float(self._val)
            return _Param(self._params.get(name))

        def create_publisher(self, msg_type, topic, qos):
            class _Pub:
                def __init__(self):
                    self.published = []
                def publish(self, msg):
                    self.published.append(msg)
            return _Pub()

        def create_timer(self, period, callback):
            return None

        def get_logger(self):
            return self._logger

        def get_clock(self):
            class _Clock:
                def now(self):
                    class _Now:
                        def to_msg(self):
                            t = time.time()
                            sec = int(t)
                            nanosec = int((t - sec) * 1e9)
                            return TimeMsg(sec=sec, nanosec=nanosec)
                    return _Now()
            return _Clock()

        def destroy_node(self):
            pass


HOST = "0.0.0.0"
PORT = 5000
MAX_QUEUE_SIZE = 4
MAX_RGB_DEPTH_SKEW_MS = 33.0


def recv_exact(conn: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(min(1024 * 1024, size - len(data)))
        if not chunk:
            raise ConnectionError("Sender disconnected")
        data.extend(chunk)
    return bytes(data)


class D455Receiver(Node):
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        queue_size: Optional[int] = None,
        max_skew_ms: Optional[float] = None,
    ):
        super().__init__("d455_receiver")

        self.declare_parameter("host", HOST)
        self.declare_parameter("port", PORT)
        self.declare_parameter("queue_size", MAX_QUEUE_SIZE)
        self.declare_parameter("max_skew_ms", MAX_RGB_DEPTH_SKEW_MS)

        self.host = host if host is not None else self.get_parameter("host").get_parameter_value().string_value
        self.port = port if port is not None else self.get_parameter("port").get_parameter_value().integer_value
        self.queue_size = queue_size if queue_size is not None else self.get_parameter("queue_size").get_parameter_value().integer_value
        self.max_skew_ms = max_skew_ms if max_skew_ms is not None else self.get_parameter("max_skew_ms").get_parameter_value().double_value

        self.rgb_pub = self.create_publisher(Image, "/camera/camera/color/image_raw", 2)
        self.depth_pub = self.create_publisher(Image, "/camera/camera/aligned_depth_to_color/image_raw", 2)
        self.camera_info_pub = self.create_publisher(CameraInfo, "/camera/camera/color/camera_info", 2)
        self.imu_pub = self.create_publisher(Imu, "/camera/camera/imu", 20)

        self.frame_queue = queue.Queue(maxsize=self.queue_size)
        self.stop_event = threading.Event()
        self.active_conn: Optional[socket.socket] = None

        # Telemetry & stats
        self.total_frames_received = 0
        self.total_frames_published = 0
        self.dropped_frames = 0
        self.total_reconnections = 0
        self.last_log_time = time.monotonic()
        self.fps_frame_count = 0
        self.imu_count = 0

        # TCP Server
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        self.server.settimeout(1.0)
        self.bound_port = self.server.getsockname()[1]

        self.get_logger().info(f"D455 TCP receiver listening on {self.host}:{self.bound_port} (queue={self.queue_size})")

        # Non-blocking dedicated background socket thread
        self.rx_thread = threading.Thread(target=self._socket_worker, daemon=True)
        self.rx_thread.start()

        # High-rate dispatch timer on ROS executor
        self.dispatch_timer = self.create_timer(0.005, self._dispatch_queued_frames)

    def _socket_worker(self):
        while not self.stop_event.is_set():
            conn = None
            try:
                try:
                    conn, addr = self.server.accept()
                    self.active_conn = conn
                except (socket.timeout, OSError):
                    continue

                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
                conn.settimeout(5.0)

                self.total_reconnections += 1
                self.get_logger().info(f"Connected to D455 sender from {addr} (reconnections={self.total_reconnections})")

                while not self.stop_event.is_set():
                    try:
                        raw_len = recv_exact(conn, 4)
                        header_size = struct.unpack("!I", raw_len)[0]
                        if not 1 <= header_size <= 1024 * 1024:
                            raise ValueError(f"Invalid header size: {header_size}")

                        header_bytes = recv_exact(conn, header_size)
                        header = json.loads(header_bytes.decode("utf-8"))

                        recv_time = time.time()
                        rgb_size = header["color"]["payload_size"]
                        depth_size = header["depth"]["payload_size"]

                        rgb_bytes = recv_exact(conn, rgb_size)
                        depth_bytes = recv_exact(conn, depth_size)

                        packet = {
                            "header": header,
                            "rgb_bytes": rgb_bytes,
                            "depth_bytes": depth_bytes,
                            "receive_time": recv_time,
                        }

                        self.total_frames_received += 1

                        # Bounded latest-frame buffering
                        try:
                            self.frame_queue.put_nowait(packet)
                        except queue.Full:
                            try:
                                _ = self.frame_queue.get_nowait()
                                self.dropped_frames += 1
                            except queue.Empty:
                                pass
                            try:
                                self.frame_queue.put_nowait(packet)
                            except queue.Full:
                                self.dropped_frames += 1

                    except (socket.timeout, socket.error, ConnectionError) as ex:
                        self.get_logger().warn(f"Sender connection interrupted: {ex}. Re-entering accept loop...")
                        break

            except Exception as ex:
                if not self.stop_event.is_set():
                    self.get_logger().error(f"Socket worker error: {ex}")
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                self.active_conn = None

    def _dispatch_queued_frames(self):
        while True:
            try:
                packet = self.frame_queue.get_nowait()
            except queue.Empty:
                break

            try:
                self._publish_frame(packet)
                self.total_frames_published += 1
                self.fps_frame_count += 1
            except Exception as ex:
                self.get_logger().error(f"Error publishing frame: {ex}")

        # Periodic statistics logging every 1.0 second
        now = time.monotonic()
        dt = now - self.last_log_time
        if dt >= 1.0:
            rgb_hz = self.fps_frame_count / dt
            imu_hz = self.imu_count / dt
            q_depth = self.frame_queue.qsize()
            self.get_logger().info(
                f"[D455 Bridge] RGB: {rgb_hz:.1f} Hz | Depth: {rgb_hz:.1f} Hz | IMU: {imu_hz:.1f} Hz | "
                f"Queue: {q_depth}/{self.queue_size} | Drops: {self.dropped_frames} | Total: {self.total_frames_published}"
            )
            self.last_log_time = now
            self.fps_frame_count = 0
            self.imu_count = 0

    def _publish_frame(self, packet: Dict[str, Any]):
        header = packet["header"]
        rgb_bytes = packet["rgb_bytes"]
        depth_bytes = packet["depth_bytes"]
        receive_time = packet["receive_time"]

        color_info = header["color"]
        depth_info = header["depth"]
        dt_ms = header.get("rgb_depth_dt_ms", 0.0)

        # Reject invalid RGB-depth synchronization
        if dt_ms > self.max_skew_ms:
            self.get_logger().warn(
                f"Frame {header['frame_id']} rejected: RGB-depth skew {dt_ms:.2f}ms exceeds tolerance {self.max_skew_ms:.2f}ms"
            )
            return

        # Synchronized ROS 2 Timestamp
        now_stamp = self.get_clock().now().to_msg()

        # RGB Image
        rgb_msg = Image()
        rgb_msg.header.stamp = now_stamp
        rgb_msg.header.frame_id = "camera_color_optical_frame"
        rgb_msg.height = color_info["height"]
        rgb_msg.width = color_info["width"]
        rgb_msg.encoding = "bgr8"
        rgb_msg.step = color_info["width"] * 3
        if hasattr(rgb_msg.data, "frombytes"):
            rgb_msg.data.frombytes(rgb_bytes)
        else:
            rgb_msg.data = rgb_bytes
        self.rgb_pub.publish(rgb_msg)

        # Depth Image
        depth_msg = Image()
        depth_msg.header.stamp = now_stamp
        depth_msg.header.frame_id = "camera_color_optical_frame"
        depth_msg.height = depth_info["height"]
        depth_msg.width = depth_info["width"]
        depth_msg.encoding = "16UC1"
        depth_msg.step = depth_info["width"] * 2
        if hasattr(depth_msg.data, "frombytes"):
            depth_msg.data.frombytes(depth_bytes)
        else:
            depth_msg.data = depth_bytes
        self.depth_pub.publish(depth_msg)

        # CameraInfo
        camera_info = CameraInfo()
        camera_info.header.stamp = now_stamp
        camera_info.header.frame_id = "camera_color_optical_frame"
        camera_info.width = color_info["width"]
        camera_info.height = color_info["height"]
        camera_info.k = [
            color_info["fx"], 0.0, color_info["ppx"],
            0.0, color_info["fy"], color_info["ppy"],
            0.0, 0.0, 1.0,
        ]
        camera_info.d = color_info.get("distortion", [0.0, 0.0, 0.0, 0.0, 0.0])
        camera_info.distortion_model = color_info.get("distortion_model", "plumb_bob")
        camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info.p = [
            color_info["fx"], 0.0, color_info["ppx"], 0.0,
            0.0, color_info["fy"], color_info["ppy"], 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        self.camera_info_pub.publish(camera_info)

        # Publish IMU samples if included
        imu_samples = header.get("imu", [])
        for imu_sample in imu_samples:
            imu_msg = Imu()
            imu_msg.header.stamp = now_stamp
            imu_msg.header.frame_id = "camera_imu_optical_frame"
            accel = imu_sample.get("accel", [0.0, 0.0, 0.0])
            gyro = imu_sample.get("gyro", [0.0, 0.0, 0.0])
            imu_msg.linear_acceleration.x = float(accel[0])
            imu_msg.linear_acceleration.y = float(accel[1])
            imu_msg.linear_acceleration.z = float(accel[2])
            imu_msg.angular_velocity.x = float(gyro[0])
            imu_msg.angular_velocity.y = float(gyro[1])
            imu_msg.angular_velocity.z = float(gyro[2])
            self.imu_pub.publish(imu_msg)
            self.imu_count += 1

    def destroy_node(self):
        self.stop_event.set()
        if self.active_conn is not None:
            try:
                self.active_conn.close()
            except Exception:
                pass
        try:
            self.server.close()
        except Exception:
            pass
        if hasattr(self, "rx_thread") and self.rx_thread.is_alive():
            self.rx_thread.join(timeout=1.0)
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
