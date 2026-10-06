# ruff: noqa: BLE001, S110
"""Production-Grade Intel RealSense D455 Sender for Windows -> WSL2 Transport.
Implements Stage 1 Data Acquisition and Transport:
- Queries authoritative hardware calibration, stream profile, and depth scale directly from RealSense SDK.
- Decouples USB acquisition from TCP transmission using a bounded latest-frame buffer (capacity=1).
- Captures 6-axis IMU (accel + gyro) with hardware timestamps.
- Preserves sensor timestamps, timestamp domains, and frame counters.
- Instruments wait, alignment, serialization, and network transmission latencies with monotonic clocks.
- Handles camera disconnects and TCP dropouts with bounded, safe retry loops.
"""

from __future__ import annotations

import collections
import json
import os
import queue
import socket
import struct
import sys
import threading
import time
import uuid
from typing import Any

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


# Protected Operating Points (deliberately tuned and validated)
WIDTH: int = 640
HEIGHT: int = 480
FPS: int = 30
DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 5000
SOCKET_SNDBUF: int = 4 * 1024 * 1024  # 4 MB socket send buffer
MAX_TRANSPORT_QUEUE_CAPACITY: int = 1   # Latest-frame policy: queue age <= 33ms @ 30 FPS


def resolve_wsl_host(preferred_host: str, port: int) -> str:
    """Resolve WSL host IP directly via Hyper-V virtual switch to bypass buggy wslrelay."""
    env_host = os.environ.get("WSL_HOST")
    if env_host:
        return env_host.strip()

    if preferred_host not in ("127.0.0.1", "localhost"):
        return preferred_host

    try:
        import subprocess
        res = subprocess.run(["wsl", "hostname", "-I"], capture_output=True, text=True, timeout=2, check=False)
        if res.returncode != 0 or not res.stdout.strip():
            res = subprocess.run(["wsl", "-d", "Ubuntu-22.04", "hostname", "-I"], capture_output=True, text=True, timeout=2, check=False)
        if res.returncode == 0:
            ip = res.stdout.strip().split()[0]
            if ip:
                print(f"[D455 Sender] Auto-detected direct WSL2 Hyper-V IP: {ip}")
                return ip
    except Exception:
        pass

    return preferred_host


class RollingLatencyTracker:
    """Thread-safe, bounded rolling window for computing P50, P95, and P99 latency percentiles."""

    def __init__(self, window_size: int = 150):
        self._window_size = window_size
        self._samples: collections.deque[float] = collections.deque(maxlen=window_size)
        self._lock = threading.Lock()

    def add(self, value_ms: float) -> None:
        with self._lock:
            self._samples.append(float(value_ms))

    def percentiles(self) -> tuple[float, float, float]:
        with self._lock:
            if not self._samples:
                return (0.0, 0.0, 0.0)
            arr = np.sort(list(self._samples))
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
            return (p50, p95, p99)


def extract_frame_metadata(frame) -> dict[str, Any]:
    """Safely extracts hardware-reported metadata using the supports_frame_metadata pattern."""
    metadata: dict[str, Any] = {}
    if rs is None or frame is None:
        return metadata

    query_fields = [
        ("frame_counter", rs.frame_metadata_value.frame_counter),
        ("sensor_timestamp", rs.frame_metadata_value.sensor_timestamp),
        ("time_of_arrival", rs.frame_metadata_value.time_of_arrival),
        ("actual_fps", rs.frame_metadata_value.actual_fps),
        ("actual_exposure", rs.frame_metadata_value.actual_exposure),
        ("gain", rs.frame_metadata_value.gain),
        ("backend_timestamp", rs.frame_metadata_value.backend_timestamp),
    ]

    for field_name, rs_val in query_fields:
        try:
            if frame.supports_frame_metadata(rs_val):
                metadata[field_name] = int(frame.get_frame_metadata(rs_val))
            else:
                metadata[field_name] = None
        except Exception:
            metadata[field_name] = None

    return metadata


def create_sender_pipeline(enable_imu: bool = True):
    """Initializes RealSense D455 pipeline and queries runtime calibration and scale."""
    if rs is None:
        raise ImportError("pyrealsense2 is required to stream from Intel RealSense D455.")

    pipeline = rs.pipeline()
    config = rs.config()

    # Protected 640x480 @ 30 FPS configuration
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    imu_enabled = False
    if enable_imu:
        try:
            config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
            config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
            imu_enabled = True
        except Exception as ex:
            print(f"[D455 Sender] Motion streams unavailable: {ex}")

    profile = pipeline.start(config)

    # Runtime depth sensor depth scale (Authoritative: do NOT hardcode 1000.0)
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    # Factory color intrinsics
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_profile.get_intrinsics()

    intrinsics_dict = {
        "fx": float(color_intr.fx),
        "fy": float(color_intr.fy),
        "ppx": float(color_intr.ppx),
        "ppy": float(color_intr.ppy),
        "distortion": [float(c) for c in color_intr.coeffs],
        "distortion_model": str(color_intr.model).replace("distortion.", ""),
    }

    # Device properties
    dev_info = {
        "name": device.get_info(rs.camera_info.name) if device.supports(rs.camera_info.name) else "Intel RealSense D455",
        "serial_number": device.get_info(rs.camera_info.serial_number) if device.supports(rs.camera_info.serial_number) else "unknown",
        "firmware_version": device.get_info(rs.camera_info.firmware_version) if device.supports(rs.camera_info.firmware_version) else "unknown",
        "usb_type": device.get_info(rs.camera_info.usb_type_descriptor) if device.supports(rs.camera_info.usb_type_descriptor) else "unknown",
    }

    align = rs.align(rs.stream.color)
    return pipeline, align, depth_scale, intrinsics_dict, imu_enabled, dev_info


class D455SenderService:
    """Manages independent acquisition and TCP transmission workers for the D455."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self.stop_event = threading.Event()

        # Bounded latest-frame queue: capacity 1 guarantees zero buffer age accumulation
        self.transport_queue: queue.Queue = queue.Queue(maxsize=MAX_TRANSPORT_QUEUE_CAPACITY)

        # Telemetry & Latency trackers (monotonic time)
        self.wait_tracker = RollingLatencyTracker()
        self.align_tracker = RollingLatencyTracker()
        self.send_tracker = RollingLatencyTracker()
        self.skew_tracker = RollingLatencyTracker()

        self.frames_captured = 0
        self.frames_sent = 0
        self.frames_dropped_transport = 0
        self.reconnect_count = 0
        self.session_id = f"d455_{uuid.uuid4().hex[:8]}_{int(time.time())}"

    def send_packet(self, sock: socket.socket, header: dict[str, Any], rgb_bytes: bytes, depth_bytes: bytes) -> float:
        """Serializes and sends a Stage 1 packet over TCP socket with monotonic duration timing."""
        t_start = time.perf_counter()
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        payload = b"".join([
            struct.pack("!I", len(header_bytes)),
            header_bytes,
            rgb_bytes,
            depth_bytes,
        ])
        sock.sendall(payload)
        duration_ms = (time.perf_counter() - t_start) * 1000.0
        return duration_ms

    def capture_loop(self):
        """Dedicated acquisition thread: never blocks behind network I/O."""
        pipeline = None
        align = None

        while not self.stop_event.is_set():
            if pipeline is None:
                try:
                    pipeline, align, depth_scale, intrinsics, imu_enabled, dev_info = create_sender_pipeline(enable_imu=True)
                    self.session_id = f"d455_{uuid.uuid4().hex[:8]}_{int(time.time())}"
                    print(f"[D455 Sender] RealSense pipeline started (session={self.session_id}).")
                    print(f"[D455 Sender] Device: {dev_info['name']} (FW: {dev_info['firmware_version']}, USB: {dev_info['usb_type']})")
                    print(f"[D455 Sender] Runtime depth scale: {depth_scale:.6f} m/unit | Intrinsics: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")
                except Exception as ex:
                    print(f"[D455 Sender] Waiting for D455 USB connection: {ex}. Retrying in 2.0s...")
                    time.sleep(2.0)
                    continue

            try:
                t0_wait = time.perf_counter()
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                wait_ms = (time.perf_counter() - t0_wait) * 1000.0
                self.wait_tracker.add(wait_ms)

                t0_align = time.perf_counter()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                align_ms = (time.perf_counter() - t0_align) * 1000.0
                self.align_tracker.add(align_ms)

                if not color_frame or not depth_frame:
                    continue

                color_data = np.asanyarray(color_frame.get_data())
                depth_data = np.asanyarray(depth_frame.get_data())

                color_ts_raw = color_frame.get_timestamp()
                depth_ts_raw = depth_frame.get_timestamp()
                # RealSense get_timestamp() returns milliseconds -> convert to float seconds
                color_ts_sec = color_ts_raw * 1e-3
                depth_ts_sec = depth_ts_raw * 1e-3
                skew_ms = abs(color_ts_raw - depth_ts_raw)
                self.skew_tracker.add(skew_ms)

                color_domain = str(color_frame.get_frame_timestamp_domain()).replace("timestamp_domain.", "").lower()
                depth_domain = str(depth_frame.get_frame_timestamp_domain()).replace("timestamp_domain.", "").lower()

                # Extract optional hardware metadata
                color_meta = extract_frame_metadata(color_frame)
                depth_meta = extract_frame_metadata(depth_frame)

                # Collect high-rate IMU if present
                imu_samples = []
                if imu_enabled:
                    accel_f = frames.first_or_default(rs.stream.accel)
                    gyro_f = frames.first_or_default(rs.stream.gyro)
                    if accel_f and gyro_f:
                        accel_d = accel_f.as_motion_frame().get_motion_data()
                        gyro_d = gyro_f.as_motion_frame().get_motion_data()
                        imu_samples.append({
                            "timestamp": accel_f.get_timestamp() * 1e-3,
                            "domain": str(accel_f.get_frame_timestamp_domain()).replace("timestamp_domain.", "").lower(),
                            "accel": [float(accel_d.x), float(accel_d.y), float(accel_d.z)],
                            "gyro": [float(gyro_d.x), float(gyro_d.y), float(gyro_d.z)],
                        })

                # Construct Protocol v2 Transport Header
                header = {
                    "protocol_version": 2,
                    "type": "d455_rgbd",
                    "session_id": self.session_id,
                    "frame_id": self.frames_captured,
                    "host_monotonic_time": time.perf_counter(),
                    "host_wall_time": time.time(),
                    "color": {
                        "timestamp": color_ts_sec,
                        "timestamp_raw_ms": color_ts_raw,
                        "timestamp_domain": color_domain,
                        "width": WIDTH,
                        "height": HEIGHT,
                        "format": "bgr8",
                        "channels": 3,
                        "bytes_per_channel": 1,
                        "payload_size": color_data.nbytes,
                        "metadata": color_meta,
                        **intrinsics,
                    },
                    "depth": {
                        "timestamp": depth_ts_sec,
                        "timestamp_raw_ms": depth_ts_raw,
                        "timestamp_domain": depth_domain,
                        "width": WIDTH,
                        "height": HEIGHT,
                        "format": "z16",
                        "channels": 1,
                        "bytes_per_channel": 2,
                        "payload_size": depth_data.nbytes,
                        "depth_scale": depth_scale,
                        "is_aligned_to_color": True,
                        "metadata": depth_meta,
                    },
                    "imu": imu_samples,
                    "rgb_depth_dt_ms": skew_ms,
                    "device": dev_info,
                }

                item = (header, color_data.tobytes(), depth_data.tobytes())

                # Non-blocking bounded push: drops stale video frame if transport thread is behind,
                # while preserving all accumulated IMU telemetry in the replacement packet.
                try:
                    self.transport_queue.put_nowait(item)
                except queue.Full:
                    try:
                        stale_item = self.transport_queue.get_nowait()
                        self.frames_dropped_transport += 1
                        stale_imu = stale_item[0].get("imu", [])
                        if stale_imu:
                            # Prepend IMU from dropped video frame so high-rate state estimation is never interrupted
                            item[0]["imu"] = stale_imu + item[0].get("imu", [])
                    except queue.Empty:
                        pass
                    try:
                        self.transport_queue.put_nowait(item)
                    except queue.Full:
                        self.frames_dropped_transport += 1

                self.frames_captured += 1

            except Exception as ex:
                print(f"[D455 Sender] RealSense acquisition fault ({ex}). Restarting pipeline...")
                if pipeline:
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
                    pipeline = None
                time.sleep(1.0)

        if pipeline:
            try:
                pipeline.stop()
            except Exception:
                pass

    def transport_loop(self):
        """Dedicated transport thread: handles TCP connection, framing, and socket I/O."""
        sock: socket.socket | None = None
        last_log_time = time.monotonic()
        fps_counter = 0

        while not self.stop_event.is_set():
            if sock is None:
                try:
                    target_host = resolve_wsl_host(self.host, self.port)
                    print(f"[D455 Sender] Connecting to receiver at {target_host}:{self.port}...")
                    sock = socket.create_connection((target_host, self.port), timeout=3.0)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCKET_SNDBUF)
                    sock.settimeout(None)
                    self.reconnect_count += 1
                    print(f"[D455 Sender] Connected successfully (session={self.session_id}, reconnects={self.reconnect_count}).")
                except OSError:
                    time.sleep(1.0)
                    continue

            try:
                item = self.transport_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            header, rgb_bytes, depth_bytes = item

            try:
                send_ms = self.send_packet(sock, header, rgb_bytes, depth_bytes)
                self.send_tracker.add(send_ms)
                self.frames_sent += 1
                fps_counter += 1
            except (OSError, BrokenPipeError, ConnectionResetError) as exc:
                print(f"[D455 Sender] Socket transmission lost ({exc}). Entering reconnect loop...")
                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass
                sock = None
                continue

            # Periodic Structured Telemetry (every 1.0 second)
            now_mono = time.monotonic()
            dt = now_mono - last_log_time
            if dt >= 1.0:
                cur_fps = fps_counter / dt
                w_p50, w_p95, _ = self.wait_tracker.percentiles()
                a_p50, a_p95, _ = self.align_tracker.percentiles()
                s_p50, s_p95, _ = self.send_tracker.percentiles()
                sk_p50, _, _ = self.skew_tracker.percentiles()

                print(
                    f"[D455 Sender] {cur_fps:.1f} FPS | "
                    f"Wait: {w_p50:.1f}ms (P95={w_p95:.1f}ms) | "
                    f"Align: {a_p50:.1f}ms (P95={a_p95:.1f}ms) | "
                    f"Send: {s_p50:.1f}ms (P95={s_p95:.1f}ms) | "
                    f"Skew: {sk_p50:.1f}ms | Drops: {self.frames_dropped_transport}"
                )
                last_log_time = now_mono
                fps_counter = 0

        if sock:
            try:
                sock.close()
            except Exception:
                pass


def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    service = D455SenderService(host=host, port=port)

    capture_t = threading.Thread(target=service.capture_loop, name="D455CaptureWorker", daemon=True)
    transport_t = threading.Thread(target=service.transport_loop, name="D455TransportWorker", daemon=True)

    capture_t.start()
    transport_t.start()

    print("[D455 Sender] Stage 1 Sender running. Press Ctrl+C to terminate.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[D455 Sender] Shutting down cleanly...")
    finally:
        service.stop_event.set()
        capture_t.join(timeout=2.0)
        transport_t.join(timeout=2.0)
        print("[D455 Sender] Shutdown complete.")


if __name__ == "__main__":
    target_host = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HOST
    target_port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT
    main(target_host, target_port)
