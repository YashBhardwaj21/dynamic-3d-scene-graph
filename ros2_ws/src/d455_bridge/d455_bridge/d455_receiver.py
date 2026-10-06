# ruff: noqa: BLE001, S110
"""Production-Grade D455 TCP Receiver and ROS 2 Bridge Node.

Receives serialized Stage 1 D455 RGB-D and IMU frames over TCP socket from Windows host.
Applies:
- Strict protocol validation and length verification.
- ClockMapping offset and drift tracking (Windows acquisition clock -> WSL2 ROS clock).
- RGB/Depth hardware synchronization validation (50 ms tolerance).
- Bounded latest-frame buffering (drop-oldest queue policy).
- Zero-crash fault isolation for malformed packets, network dropouts, and camera resets.
"""

from __future__ import annotations

import json
import queue
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any


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


_ensure_paths()

from scene_graph.data.timestamp import ClockMapping, TimestampDomain


try:
    import rclpy
    from builtin_interfaces.msg import Time as TimeMsg
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image, Imu
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
            def warning(self, msg): pass
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


# Protected Operating Points
HOST = "0.0.0.0"
PORT = 5000
MAX_QUEUE_SIZE = 2  # Low-latency latest-frame policy: max 2 frames in flight
MAX_RGB_DEPTH_SKEW_MS = 50.0  # Protected live sync tolerance (0.05 s)
MAX_HEADER_SIZE_BYTES = 1024 * 1024       # 1 MB maximum JSON header
MAX_STREAM_PAYLOAD_BYTES = 10 * 1024 * 1024 # 10 MB maximum payload per modality


def recv_exact(conn: socket.socket, size: int) -> bytes:
    """Reads exactly `size` bytes from socket, raising ConnectionError on premature close."""
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(min(1024 * 1024, size - len(data)))
        if not chunk:
            raise ConnectionError("Sender closed TCP connection prematurely")
        data.extend(chunk)
    return bytes(data)


class D455Receiver(Node):
    """ROS 2 Node bridging TCP-streamed RealSense D455 sensor data into ROS topics."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        queue_size: int | None = None,
        max_skew_ms: float | None = None,
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

        # Publishers
        self.rgb_pub = self.create_publisher(Image, "/camera/camera/color/image_raw", 2)
        self.depth_pub = self.create_publisher(Image, "/camera/camera/aligned_depth_to_color/image_raw", 2)
        self.camera_info_pub = self.create_publisher(CameraInfo, "/camera/camera/color/camera_info", 2)
        self.imu_pub = self.create_publisher(Imu, "/camera/camera/imu", 50)

        # Bounded Queue & Workers
        self.frame_queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self.stop_event = threading.Event()
        self.active_conn: socket.socket | None = None

        # Clock Mapping & State (Learns from arrival time at socket boundary, NOT queue dispatch time)
        self.clock_mapping = ClockMapping(smoothing_alpha=0.05, max_drift_jump_sec=0.5, max_stale_duration_sec=2.0)
        self.current_session_id: str | None = None
        self.last_sequence_number: int = -1

        # Diagnostics & Metrics
        self.total_frames_received = 0
        self.total_frames_published = 0
        self.dropped_frames = 0
        self.total_reconnections = 0
        self.rejected_skew_count = 0
        self.out_of_order_count = 0
        self.duplicate_count = 0
        self.sequence_gap_count = 0
        self.missing_frames_count = 0
        self.invalid_packet_count = 0
        self.last_log_time = time.monotonic()
        self.fps_frame_count = 0
        self.imu_count = 0

        # TCP Server Setup
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        self.server.settimeout(1.0)
        self.bound_port = self.server.getsockname()[1]

        self.get_logger().info(f"D455 TCP receiver listening on {self.host}:{self.bound_port} (queue={self.queue_size}, max_skew={self.max_skew_ms}ms)")

        # Background Non-blocking Socket Thread
        self.rx_thread = threading.Thread(target=self._socket_worker, name="D455SocketWorker", daemon=True)
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
                except (TimeoutError, OSError):
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
                        # Framing validation: invalid length destroys stream alignment -> disconnect
                        if not 1 <= header_size <= MAX_HEADER_SIZE_BYTES:
                            self.invalid_packet_count += 1
                            self.get_logger().error(f"Malformed header length: {header_size} bytes. Disconnecting to realign.")
                            break

                        header_bytes = recv_exact(conn, header_size)
                        try:
                            header = json.loads(header_bytes.decode("utf-8"))
                        except Exception as json_err:
                            self.invalid_packet_count += 1
                            self.get_logger().error(f"Malformed JSON in transport header: {json_err}. Disconnecting to realign.")
                            break

                        # Exact arrival timestamp in ROS time at the network boundary (zero queue latency)
                        ros_clock = self.get_clock().now()
                        if hasattr(ros_clock, "nanoseconds"):
                            arrival_ros_sec = float(ros_clock.nanoseconds) * 1e-9
                        else:
                            t_msg = ros_clock.to_msg()
                            arrival_ros_sec = float(t_msg.sec) + float(t_msg.nanosec) * 1e-9

                        recv_time_wall = time.time()
                        recv_time_mono = time.monotonic()

                        # Invariant Validation
                        if "color" not in header or "depth" not in header:
                            self.invalid_packet_count += 1
                            self.get_logger().warn("Packet missing color or depth metadata. Skipping.")
                            continue

                        rgb_size = header["color"].get("payload_size")
                        depth_size = header["depth"].get("payload_size")

                        # Payload size bounds validation: corrupt sizes destroy stream framing -> disconnect
                        if not (isinstance(rgb_size, int) and isinstance(depth_size, int) and
                                100 <= rgb_size <= MAX_STREAM_PAYLOAD_BYTES and 100 <= depth_size <= MAX_STREAM_PAYLOAD_BYTES):
                            self.invalid_packet_count += 1
                            self.get_logger().error(f"Invalid payload bounds: rgb={rgb_size}B, depth={depth_size}B. Disconnecting to realign.")
                            break

                        # Read exact payloads from stream
                        rgb_bytes = recv_exact(conn, rgb_size)
                        depth_bytes = recv_exact(conn, depth_size)

                        # Semantic dimension validation: stream bytes are now consumed so connection remains intact
                        c_w = header["color"].get("width")
                        c_h = header["color"].get("height")
                        c_ch = header["color"].get("channels", 3)
                        c_bpc = header["color"].get("bytes_per_channel", 1)
                        d_w = header["depth"].get("width")
                        d_h = header["depth"].get("height")

                        if not (isinstance(c_w, int) and c_w > 0 and
                                isinstance(c_h, int) and c_h > 0 and
                                isinstance(c_ch, int) and c_ch > 0 and
                                isinstance(c_bpc, int) and c_bpc > 0 and
                                isinstance(d_w, int) and d_w > 0 and
                                isinstance(d_h, int) and d_h > 0):
                            self.invalid_packet_count += 1
                            self.get_logger().error("Non-positive or non-integer image dimensions in header. Discarding packet.")
                            continue

                        if rgb_size != (c_w * c_h * c_ch * c_bpc):
                            self.invalid_packet_count += 1
                            self.get_logger().error(f"Inconsistent RGB payload size {rgb_size} vs expected {c_w*c_h*c_ch*c_bpc}. Discarding packet.")
                            continue

                        if depth_size != (d_w * d_h * 2):
                            self.invalid_packet_count += 1
                            self.get_logger().error(f"Inconsistent depth payload size {depth_size} vs expected {d_w*d_h*2}. Discarding packet.")
                            continue

                        # Session Continuity Check: clear stale queued video to prevent cross-session contamination
                        session_id = str(header.get("session_id", "default"))
                        if session_id != self.current_session_id:
                            self.get_logger().info(f"New sensor session detected: '{session_id}'. Purging stale queue and resetting clock mapping.")
                            cleared_stale = 0
                            while True:
                                try:
                                    self.frame_queue.get_nowait()
                                    cleared_stale += 1
                                except queue.Empty:
                                    break
                            if cleared_stale > 0:
                                self.dropped_frames += cleared_stale
                                self.get_logger().info(f"Purged {cleared_stale} stale buffered frames from previous session.")
                            self.clock_mapping.reset()
                            self.current_session_id = session_id
                            self.last_sequence_number = -1

                        # Sequence Gap and Ordering Tracking
                        seq_num = int(header.get("frame_id", header.get("sequence_number", 0)))
                        if self.last_sequence_number >= 0:
                            if seq_num < self.last_sequence_number:
                                self.out_of_order_count += 1
                            elif seq_num == self.last_sequence_number:
                                self.duplicate_count += 1
                            elif seq_num > self.last_sequence_number + 1:
                                gap = seq_num - self.last_sequence_number - 1
                                self.sequence_gap_count += 1
                                self.missing_frames_count += gap
                        self.last_sequence_number = seq_num

                        # Clock Mapping: Updated directly at network arrival time (excludes queue delays)
                        color_ts = float(header["color"]["timestamp"])
                        self.clock_mapping.update(source_time=color_ts, target_time=arrival_ros_sec)

                        packet = {
                            "header": header,
                            "rgb_bytes": rgb_bytes,
                            "depth_bytes": depth_bytes,
                            "arrival_ros_sec": arrival_ros_sec,
                            "receive_time_wall": recv_time_wall,
                            "receive_time_mono": recv_time_mono,
                        }

                        self.total_frames_received += 1

                        # Bounded latest-frame buffering: drop stale frame on overflow
                        try:
                            self.frame_queue.put_nowait(packet)
                        except queue.Full:
                            try:
                                stale_pkt = self.frame_queue.get_nowait()
                                self.dropped_frames += 1
                                # Publish IMU from dropped packet to preserve complete high-rate telemetry
                                self._publish_imu_only(stale_pkt)
                            except queue.Empty:
                                pass
                            try:
                                self.frame_queue.put_nowait(packet)
                            except queue.Full:
                                self.dropped_frames += 1

                    except (TimeoutError, OSError, ConnectionError) as ex:
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
        """Latest-frame dispatch: drains queue and processes only the newest RGB-D frame.
        
        Publishes IMU telemetry for all drained frames so high-rate state estimation is never interrupted.
        """
        packets = []
        while True:
            try:
                packets.append(self.frame_queue.get_nowait())
            except queue.Empty:
                break

        if not packets:
            return

        # Drain queue: publish IMU for stale intermediate packets to guarantee complete IMU telemetry
        for pkt in packets[:-1]:
            try:
                self._publish_imu_only(pkt)
            except Exception as ex:
                self.get_logger().error(f"Error publishing IMU from drained packet: {ex}")
            self.dropped_frames += 1

        # Publish full RGB-D + CameraInfo + IMU for the single newest frame
        try:
            self._publish_frame(packets[-1])
            self.total_frames_published += 1
            self.fps_frame_count += 1
        except Exception as ex:
            self.get_logger().error(f"Error publishing frame: {ex}")

        # Periodic telemetry summary
        now = time.monotonic()
        dt = now - self.last_log_time
        if dt >= 1.0:
            rgb_hz = self.fps_frame_count / dt
            imu_hz = self.imu_count / dt
            q_depth = self.frame_queue.qsize()
            clk_status = "HEALTHY" if self.clock_mapping.is_healthy else "UNHEALTHY"
            offset_ms = (self.clock_mapping.offset or 0.0) * 1000.0
            drift_ppm = (self.clock_mapping.last_drift or 0.0) * 1e6

            self.get_logger().info(
                f"[D455 Bridge] RGB: {rgb_hz:.1f}Hz | Depth: {rgb_hz:.1f}Hz | IMU: {imu_hz:.1f}Hz | "
                f"Queue: {q_depth}/{self.queue_size} | Drops: {self.dropped_frames} | "
                f"Gaps: {self.sequence_gap_count} (lost={self.missing_frames_count}) | "
                f"Dup: {self.duplicate_count} | OoO: {self.out_of_order_count} | SkewRej: {self.rejected_skew_count} | "
                f"Clock: {clk_status} (offset={offset_ms:.1f}ms, drift={drift_ppm:.1f}ppm)"
            )
            self.last_log_time = now
            self.fps_frame_count = 0
            self.imu_count = 0

    def _publish_imu_only(self, packet: dict[str, Any]):
        """Publishes all IMU samples from a packet whose RGB-D frame was skipped/dropped."""
        header = packet["header"]
        color_ts = float(header["color"]["timestamp"])
        fallback_sec = packet.get("arrival_ros_sec", time.time())
        imu_samples = header.get("imu", [])
        for imu_sample in imu_samples:
            imu_s_ts = float(imu_sample.get("timestamp", color_ts))
            if self.clock_mapping.is_healthy:
                imu_mapped_sec = self.clock_mapping.map_timestamp(imu_s_ts)
            else:
                imu_mapped_sec = fallback_sec

            i_sec = int(imu_mapped_sec)
            i_nanosec = int((imu_mapped_sec - i_sec) * 1e9)

            imu_msg = Imu()
            imu_msg.header.stamp = TimeMsg(sec=i_sec, nanosec=i_nanosec)
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

    def _publish_frame(self, packet: dict[str, Any]):
        header = packet["header"]
        rgb_bytes = packet["rgb_bytes"]
        depth_bytes = packet["depth_bytes"]

        color_info = header["color"]
        depth_info = header["depth"]
        color_ts = float(color_info["timestamp"])
        depth_ts = float(depth_info["timestamp"])

        # Live RGB-Depth Skew Check (Protected Operating Point: 50 ms)
        dt_ms = header.get("rgb_depth_dt_ms", abs(color_ts - depth_ts) * 1000.0)
        if dt_ms > self.max_skew_ms:
            self.rejected_skew_count += 1
            self.get_logger().warn(
                f"Frame {header.get('frame_id')} rejected: RGB-depth skew {dt_ms:.2f}ms exceeds tolerance {self.max_skew_ms:.2f}ms"
            )
            return

        # Timing Projection: Map source timestamp using calibrated network-boundary offset
        if self.clock_mapping.is_healthy:
            mapped_ts_sec = self.clock_mapping.map_timestamp(color_ts)
        else:
            mapped_ts_sec = packet.get("arrival_ros_sec", time.time())

        sec = int(mapped_ts_sec)
        nanosec = int((mapped_ts_sec - sec) * 1e9)
        mapped_stamp = TimeMsg(sec=sec, nanosec=nanosec)

        # Publish RGB Image
        rgb_msg = Image()
        rgb_msg.header.stamp = mapped_stamp
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

        # Publish Depth Image
        depth_msg = Image()
        depth_msg.header.stamp = mapped_stamp
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

        # Publish CameraInfo with factory intrinsics & distortion model
        camera_info = CameraInfo()
        camera_info.header.stamp = mapped_stamp
        camera_info.header.frame_id = "camera_color_optical_frame"
        camera_info.width = color_info["width"]
        camera_info.height = color_info["height"]
        camera_info.k = [
            color_info["fx"], 0.0, color_info["ppx"],
            0.0, color_info["fy"], color_info["ppy"],
            0.0, 0.0, 1.0,
        ]
        camera_info.d = [float(c) for c in color_info.get("distortion", [0.0, 0.0, 0.0, 0.0, 0.0])]
        camera_info.distortion_model = str(color_info.get("distortion_model", "plumb_bob"))
        camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info.p = [
            color_info["fx"], 0.0, color_info["ppx"], 0.0,
            0.0, color_info["fy"], color_info["ppy"], 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        self.camera_info_pub.publish(camera_info)

        # Publish IMU samples with individual mapped timestamps
        self._publish_imu_only(packet)

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
