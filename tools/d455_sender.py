import json
import socket
import struct
import time
from typing import Optional, List, Dict, Any
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


WIDTH = 640
HEIGHT = 480
FPS = 30
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000


def resolve_wsl_host(preferred_host: str, port: int) -> str:
    """Resolve WSL host IP directly via Hyper-V virtual switch to bypass buggy wslrelay."""
    if preferred_host not in ("127.0.0.1", "localhost"):
        return preferred_host

    try:
        import subprocess
        res = subprocess.run(["wsl", "-d", "Ubuntu-22.04", "hostname", "-I"], capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            ip = res.stdout.strip().split()[0]
            if ip:
                print(f"[D455 Sender] Auto-detected direct WSL2 Hyper-V IP: {ip}")
                return ip
    except Exception:
        pass

    return preferred_host


def create_sender_pipeline(enable_imu: bool = True):
    if rs is None:
        raise ImportError("pyrealsense2 is required to stream from Intel RealSense D455.")

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    imu_enabled = False
    if enable_imu:
        try:
            config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
            config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
            imu_enabled = True
        except Exception as ex:
            print(f"[D455] Warning: Motion streams not available: {ex}")

    profile = pipeline.start(config)

    # Actual RealSense color-to-depth alignment
    align = rs.align(rs.stream.color)

    # Extract depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Extract color intrinsics
    color_profile = profile.get_stream(rs.stream.color)
    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

    intrinsics_dict = {
        "fx": float(color_intrinsics.fx),
        "fy": float(color_intrinsics.fy),
        "ppx": float(color_intrinsics.ppx),
        "ppy": float(color_intrinsics.ppy),
        "distortion": [float(c) for c in color_intrinsics.coeffs],
        "distortion_model": str(color_intrinsics.model).replace("distortion.", ""),
    }

    return pipeline, align, depth_scale, intrinsics_dict, imu_enabled


def init_pipeline_with_retry(enable_imu: bool = True, retry_delay: float = 2.0):
    """Wait for RealSense D455 camera to be connected and initialize pipeline."""
    printed_wait = False
    while True:
        try:
            return create_sender_pipeline(enable_imu=enable_imu)
        except Exception as e:
            if not printed_wait:
                print(f"[D455 Sender] Waiting for RealSense D455 camera via USB... Plug in camera anytime. ({e})")
                printed_wait = True
            time.sleep(retry_delay)


def send_packet(sock: socket.socket, header: Dict[str, Any], rgb_bytes: bytes, depth_bytes: bytes) -> float:
    t_start = time.monotonic()
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload = b"".join([
        struct.pack("!I", len(header_bytes)),
        header_bytes,
        rgb_bytes,
        depth_bytes,
    ])
    sock.sendall(payload)
    return (time.monotonic() - t_start) * 1000.0


def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    host = resolve_wsl_host(host, port)
    print(f"[D455 Sender] Target receiver address: {host}:{port}")
    print(f"[D455 Sender] Initializing RealSense D455 camera ({WIDTH}x{HEIGHT} @ {FPS}fps)...")
    pipeline, align, depth_scale, intrinsics, imu_available = init_pipeline_with_retry(enable_imu=True)
    print(f"[D455 Sender] RealSense pipeline started. Depth scale: {depth_scale:.6f} m/unit.")
    print(f"[D455 Sender] Color intrinsics: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}, cx={intrinsics['ppx']:.1f}, cy={intrinsics['ppy']:.1f}")

    frame_id = 0
    sock: Optional[socket.socket] = None

    try:
        while True:
            # Reconnect loop if socket is not connected
            if sock is None:
                try:
                    host = resolve_wsl_host(host, port)
                    print(f"[D455 Sender] Connecting to receiver at {host}:{port}...")
                    sock = socket.create_connection((host, port), timeout=3.0)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
                    sock.settimeout(None)
                    print(f"[D455 Sender] Connected successfully to {host}:{port}.")
                except (socket.error, OSError) as exc:
                    print(f"[D455 Sender] Connection failed ({exc}). Retrying in 1.0s...")
                    time.sleep(1.0)
                    continue

            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except Exception as ex:
                print(f"[D455 Sender] Frame timeout/drop: {ex}")
                continue

            # Actual RealSense hardware alignment of depth to color stream
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_data = np.asanyarray(color_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())

            color_ts = color_frame.get_timestamp()
            depth_ts = depth_frame.get_timestamp()
            dt_ms = abs(color_ts - depth_ts)

            # Collect IMU samples if available
            imu_samples = []
            if imu_available:
                accel_frame = frames.first_or_default(rs.stream.accel)
                gyro_frame = frames.first_or_default(rs.stream.gyro)
                if accel_frame and gyro_frame:
                    accel_data = accel_frame.as_motion_frame().get_motion_data()
                    gyro_data = gyro_frame.as_motion_frame().get_motion_data()
                    imu_samples.append({
                        "timestamp": accel_frame.get_timestamp() * 1e-3,
                        "accel": [float(accel_data.x), float(accel_data.y), float(accel_data.z)],
                        "gyro": [float(gyro_data.x), float(gyro_data.y), float(gyro_data.z)],
                    })

            header = {
                "type": "d455_rgbd",
                "frame_id": frame_id,
                "host_time": time.time(),
                "color": {
                    "timestamp": color_ts,
                    "timestamp_domain": str(color_frame.get_frame_timestamp_domain()),
                    "width": WIDTH,
                    "height": HEIGHT,
                    "format": "bgr8",
                    "channels": 3,
                    "bytes_per_channel": 1,
                    "payload_size": color_data.nbytes,
                    **intrinsics,
                },
                "depth": {
                    "timestamp": depth_ts,
                    "timestamp_domain": str(depth_frame.get_frame_timestamp_domain()),
                    "width": WIDTH,
                    "height": HEIGHT,
                    "format": "z16",
                    "channels": 1,
                    "bytes_per_channel": 2,
                    "payload_size": depth_data.nbytes,
                    "depth_scale": depth_scale,
                    "is_aligned_to_color": True,
                },
                "imu": imu_samples,
                "rgb_depth_dt_ms": dt_ms,
            }

            try:
                send_ms = send_packet(sock, header, color_data.tobytes(), depth_data.tobytes())
            except (socket.error, BrokenPipeError, ConnectionResetError) as exc:
                print(f"[D455 Sender] Socket error during send ({exc}). Reconnecting...")
                if sock:
                    sock.close()
                sock = None
                continue

            frame_id += 1
            if frame_id == 1 or frame_id % 30 == 0:
                print(f"[D455 Sender] Frame {frame_id:05d} | RGB={color_data.nbytes:,}B | Depth={depth_data.nbytes:,}B | skew={dt_ms:.2f}ms | send={send_ms:.1f}ms")

    except KeyboardInterrupt:
        print("\n[D455 Sender] Shutting down cleanly...")
    finally:
        if pipeline is not None:
            pipeline.stop()
        if sock is not None:
            sock.close()
        print("[D455 Sender] Pipeline stopped and sockets closed.")


if __name__ == "__main__":
    import sys
    target_host = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HOST
    target_port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT
    main(target_host, target_port)
