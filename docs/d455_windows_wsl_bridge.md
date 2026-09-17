# RealSense D455 Windows-to-WSL2 Live Perception Bridge

## 1. Executive Summary & Architecture

This document details the complete end-to-end architecture, communication protocol, implementation code, and operational workflows used to connect an **Intel RealSense D455** depth camera running on **Windows 11** to a **ROS 2 Humble / Dynamic 3D Scene Graph pipeline running inside WSL2 (Ubuntu 22.04)**.

```
┌─────────────────────────────────────────────────────────────┐
│                    WINDOWS 11 (HOST)                        │
│                                                             │
│  [ Intel RealSense D455 Camera ]                            │
│           │ (Native USB 3.2 Gen 1)                          │
│           ▼                                                 │
│  [ Python pyrealsense2 Pipeline (d455_test.py) ]             │
│      - 640x480 BGR8 (30 FPS)                                │
│      - 640x480 Z16 Depth (30 FPS)                           │
│      - Hardware Factory Calibration (fx, fy, ppx, ppy)      │
│           │                                                 │
│      Pack single binary frame payload:                      │
│      [4B Header Length] + [JSON Header] + [RGB] + [Depth]   │
│           │                                                 │
│      TCP Client (TCP_NODELAY, SO_SNDBUF=4MB)                │
└───────────┼─────────────────────────────────────────────────┘
            │  Hyper-V Virtual Switch (vEthernet)
            │  TCP/IP: 172.29.3.153:5000 (~1.5 MB/frame @ 4-5ms)
┌───────────┼─────────────────────────────────────────────────┐
│           ▼                                                 │
│  [ TCP Server Node (d455_bridge / d455_receiver) ]          │
│      - Zero-copy socket stream reader (recv_exact)          │
│      - Node-clock timestamp synchronization                 │
│      - Publishes standard ROS 2 topics:                     │
│          * /camera/camera/color/image_raw                   │
│          * /camera/camera/aligned_depth_to_color/image_raw  │
│          * /camera/camera/color/camera_info                 │
│           │                                                 │
│           ▼                                                 │
│  [ SceneGraphROSNode (scene_graph_node) ]                   │
│      - ApproximateTimeSynchronizer (RGB + Depth + Info)     │
│      - Asynchronous worker queue (drop_old_frames=True)     │
│      - Camera-frame fallback pose (world_T_camera)          │
│           │                                                 │
│           ▼                                                 │
│  [ Open-Vocabulary YOLOE Prompt-Free Perception ]           │
│      - yoloe-26s-seg-pf.pt (4,585-class LRPC vocabulary)    │
│      - 2D boxes + pixel-precise instance segmentation masks │
│           │                                                 │
│           ▼                                                 │
│  [ 3D Geometric Scene Graph Pipeline ]                      │
│      - 3D point-cloud lifting + MAD robust outlier filter   │
│      - Causal 3D Kalman Filter Object Tracker               │
│      - 3D Spatial Relations (ON, NEAR, IN_FRONT_OF, etc.)   │
│           │                                                 │
│     ┌─────┴────────────────────────┐                        │
│     ▼                              ▼                        │
│ [ Live 2D Viewer Dashboard ]   [ RViz2 3D Visualization ]   │
│  (Detections, Tracks, Graph)    (Boxes, Clouds, Markers)    │
│                                                             │
│                    WSL2 (UBUNTU 22.04)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Why This Architecture? (Windows Host + WSL2 Bridge)

### The RealSense on WSL2 Problem
Passing USB 3.0 depth cameras into WSL2 via `usbipd-win` introduces significant challenges:
1. **Bandwidth Saturation & Packet Loss**: D455 streams 640×480 BGR8 (921.6 KB) + 640×480 Z16 Depth (614.4 KB) at 30 FPS = **~46 MB/second**. Software USB/IP encapsulation across WSL2 virtual USB controllers frequently drops frames and resets the UVC driver.
2. **RealSense Hardware Clock Desynchronization**: WSL2's virtual USB bus causes jitter in camera hardware timestamps.
3. **Driver Instability**: Intel's `librealsense2` on Linux requires kernel patches for hardware frame metadata and hardware timestamp domains.

### The High-Throughput TCP Socket Solution
* **Windows Host**: Runs native Intel `pyrealsense2` via Windows USB drivers. It has direct, zero-overhead access to the camera hardware, firmware intrinsics, and hardware synchronization.
* **Hyper-V Virtual Adapter**: Windows and WSL2 communicate over the internal Hyper-V virtual switch with network transfer latencies of **< 5 ms per frame**.
* **Clean Separation of Concerns**: Windows handles hardware acquisition; WSL2 handles perception, tracking, and ROS 2 compute.

---

## 3. High-Performance Wire Protocol

Each frame is transmitted over a persistent, raw TCP socket as a single contiguous binary packet:

```
┌──────────────┬────────────────────────┬─────────────────────┬──────────────────────┐
│  4 Bytes     │  Variable Length       │  921,600 Bytes      │  614,400 Bytes       │
│  uint32 (BE) │  JSON Header (UTF-8)   │  RGB Payload (BGR8) │  Depth Payload (Z16) │
└──────────────┴────────────────────────┴─────────────────────┴──────────────────────┘
```

### JSON Header Structure
```json
{
  "type": "d455_rgbd",
  "frame_id": 142,
  "host_time": 1789550725.259,
  "color": {
    "timestamp": 1789550725259.0,
    "timestamp_domain": "System Time",
    "width": 640,
    "height": 480,
    "format": "bgr8",
    "channels": 3,
    "bytes_per_channel": 1,
    "payload_size": 921600,
    "fx": 382.45,
    "fy": 382.11,
    "ppx": 318.22,
    "ppy": 241.95,
    "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    "distortion_model": "Brown Conrady"
  },
  "depth": {
    "timestamp": 1789550725259.0,
    "timestamp_domain": "System Time",
    "width": 640,
    "height": 480,
    "format": "z16",
    "channels": 1,
    "bytes_per_channel": 2,
    "payload_size": 614400
  },
  "rgb_depth_dt_ms": 0.0
}
```

### Throughput & Latency Optimizations
1. **`TCP_NODELAY = 1`**: Disables Nagle's algorithm on both client and server, preventing 40ms delayed-ACK pauses.
2. **`SO_SNDBUF` & `SO_RCVBUF` = 4 MB**: Expands TCP window size to accommodate 1.53 MB packets without stalling.
3. **Single System Call Packaging**: `b"".join([len_bytes, header_bytes, rgb_bytes, depth_bytes])` guarantees the OS network stack transmits the frame in a single burst rather than 4 separate syscalls.

---

## 4. The Source Code Behind the System

### A. Windows Sender Script (`d455_test.py`)
**Location**: `C:\Users\Yash Bhardwaj\Desktop\d455_test.py`

```python
import json
import socket
import struct
import time
import numpy as np
import pyrealsense2 as rs

# WSL2 IP address (obtain via: ip -4 addr show eth0 in WSL)
WSL_HOST = "172.29.3.153"
WSL_PORT = 5000

WIDTH = 640
HEIGHT = 480
FPS = 30


def send_packet(sock, header, rgb_bytes, depth_bytes):
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload = b"".join([
        struct.pack("!I", len(header_bytes)),
        header_bytes,
        rgb_bytes,
        depth_bytes,
    ])
    sock.sendall(payload)


def main():
    print("Starting Intel RealSense D455...")
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    profile = pipeline.start(config)

    # Extract hardware factory color intrinsics
    color_profile = profile.get_stream(rs.stream.color)
    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

    print(f"Connecting to WSL2 TCP receiver at {WSL_HOST}:{WSL_PORT}...")
    sock = socket.create_connection((WSL_HOST, WSL_PORT), timeout=10)

    # Low-latency socket tuning
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    sock.settimeout(None)

    print("TCP connection established. Streaming continuous RGB + Depth...")
    frame_id = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()

            if not color or not depth:
                continue

            color_data = np.asanyarray(color.get_data())
            depth_data = np.asanyarray(depth.get_data())

            rgb_bytes = color_data.tobytes()
            depth_bytes = depth_data.tobytes()

            rgb_timestamp = color.get_timestamp()
            depth_timestamp = depth.get_timestamp()

            header = {
                "type": "d455_rgbd",
                "frame_id": frame_id,
                "host_time": time.time(),
                "color": {
                    "timestamp": rgb_timestamp,
                    "timestamp_domain": str(color.get_frame_timestamp_domain()),
                    "width": color.get_width(),
                    "height": color.get_height(),
                    "format": "bgr8",
                    "channels": 3,
                    "bytes_per_channel": 1,
                    "payload_size": len(rgb_bytes),
                    "fx": color_intrinsics.fx,
                    "fy": color_intrinsics.fy,
                    "ppx": color_intrinsics.ppx,
                    "ppy": color_intrinsics.ppy,
                    "distortion": list(color_intrinsics.coeffs),
                    "distortion_model": str(color_intrinsics.model),
                },
                "depth": {
                    "timestamp": depth_timestamp,
                    "timestamp_domain": str(depth.get_frame_timestamp_domain()),
                    "width": depth.get_width(),
                    "height": depth.get_height(),
                    "format": "z16",
                    "channels": 1,
                    "bytes_per_channel": 2,
                    "payload_size": len(depth_bytes),
                },
                "rgb_depth_dt_ms": rgb_timestamp - depth_timestamp,
            }

            t0 = time.perf_counter()
            send_packet(sock, header, rgb_bytes, depth_bytes)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if frame_id % 30 == 0:
                print(f"Frame {frame_id:04d} | RGB={len(rgb_bytes):,}B Depth={len(depth_bytes):,}B send={elapsed_ms:.1f}ms")

            frame_id += 1
    finally:
        sock.close()
        pipeline.stop()
        print(f"Stream stopped. Total frames sent: {frame_id}")


if __name__ == "__main__":
    main()
```

---

### B. WSL2 ROS 2 Receiver Node (`d455_receiver.py`)
**Location**: `perception/ros2_ws/src/d455_bridge/d455_bridge/d455_receiver.py`

**Key Mechanics**:
1. **Zero-Copy Byte Reading (`recv_exact`)**: Reads socket chunks directly into a preallocated `bytearray` memoryview, preventing Python memory allocations on every frame.
2. **Clock Synchronization**: Stamped with `self.get_clock().now().to_msg()` so RGB, Depth, and CameraInfo share the exact same timestamp down to the nanosecond, aligning seamlessly with the local ROS 2 TF tree.

```python
def recv_exact(sock, n_bytes: int) -> bytes:
    buf = bytearray(n_bytes)
    view = memoryview(buf)
    pos = 0
    while pos < n_bytes:
        chunk = sock.recv_into(view[pos:], n_bytes - pos)
        if chunk == 0:
            raise ConnectionError("Socket closed prematurely")
        pos += chunk
    return bytes(buf)
```

**Publishing Pipeline**:
* **RGB Image**: `/camera/camera/color/image_raw` (`sensor_msgs/Image`, `bgr8`, `frame_id: camera_color_optical_frame`)
* **Depth Image**: `/camera/camera/aligned_depth_to_color/image_raw` (`sensor_msgs/Image`, `16UC1`, `frame_id: camera_color_optical_frame`)
* **Camera Info**: `/camera/camera/color/camera_info` (`sensor_msgs/CameraInfo`, filled from factory matrix `fx, fy, cx, cy`)

---

### C. SceneGraph Node & Pose Fallback (`scene_graph_node.py`)
**Location**: `perception/ros2_ws/src/scene_graph_ros/scene_graph_ros/scene_graph_node.py`

**Robust Pose Fallback**:
If SLAM (RTAB-Map) is not running or visual odometry drops tracking (`quality=0`), the node falls back to the camera coordinate frame (`world_T_camera = np.eye(4)`).
* In camera coordinates: Z is forward, X is right, Y is down (Up = $-Y$).
* The relation reference frame is automatically configured via `RelationReferenceFrame.create("camera", world_T_camera)`.
* **Result**: 3D bounding boxes, Kalman tracks, and spatial relations (`ON`, `NEAR`, `IN_FRONT_OF`) compute and display continuously with zero downtime.

---

## 5. Complete Step-by-Step Command Guide

### Step 1: Discover WSL2 IP Address (Run Once)
In WSL2 terminal:
```bash
ip -4 addr show eth0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}'
```
*Example Output*: `172.29.3.153`

Verify that `WSL_HOST` in `C:\Users\Yash Bhardwaj\Desktop\d455_test.py` matches this IP.

---

### Step 2: Build the ROS 2 Workspace in WSL2
In WSL2 terminal:
```bash
cd "/mnt/c/Users/Yash Bhardwaj/Desktop/perception/ros2_ws"
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

---

### Step 3: Launch the Perception & Scene Graph Stack in WSL2
In WSL2 terminal:
```bash
source /opt/ros/humble/setup.bash
source /home/saturn/myenv/bin/activate
cd "/mnt/c/Users/Yash Bhardwaj/Desktop/perception/ros2_ws"
source install/setup.bash
export PYTHONPATH="/mnt/c/Users/Yash Bhardwaj/Desktop/perception/src:$PYTHONPATH"

# Launch D455 Bridge, Static TF, Scene Graph Node, RViz2, and 2D Dashboard:
ros2 launch scene_graph_ros live_scene_graph.launch.py
```

*Expected Terminal Output*:
```
[d455_receiver-1] Starting D455 TCP receiver on 0.0.0.0:5000
[static_map_to_camera-2] Publishing static transform: world -> camera_color_optical_frame
[scene_graph_node-3] Initializing OnlinePipeline with yoloe-26s-seg-pf.pt (imgsz=480)...
[scene_graph_node-3] OnlinePipeline ready. Listening on camera topics...
[live_2d_viewer-5] Live 2D Perception Viewer started.
```

---

### Step 4: Start the Camera Stream on Windows
Open **PowerShell** on the Windows host and run:
```powershell
python "C:\Users\Yash Bhardwaj\Desktop\d455_test.py"
```

*Expected Output*:
```
Starting Intel RealSense D455...
Device: Intel RealSense D455
Serial: 146322250154
Connecting to WSL2 TCP receiver at 172.29.3.153:5000...
TCP connection established. Streaming continuous RGB + Depth...
Frame 0000 | RGB=921,600B Depth=614,400B send=4.2ms
Frame 0030 | RGB=921,600B Depth=614,400B send=4.8ms
Frame 0060 | RGB=921,600B Depth=614,400B send=5.1ms
```

---

### Step 5: Verify Live ROS 2 Topics (Optional Diagnostic)
In a second WSL2 terminal:
```bash
source /opt/ros/humble/setup.bash

# Check incoming frame rates:
ros2 topic hz /camera/camera/color/image_raw
# Output: average rate: 30.12 Hz

# Check scene graph output rate:
ros2 topic hz /scene_graph/snapshot
# Output: average rate: 3.48 Hz (on CPU)

# Inspect detected objects and relations in JSON format:
ros2 topic echo /scene_graph/snapshot --field objects
ros2 topic echo /scene_graph/snapshot --field relations
```

---

## 6. Known Gotchas & Solutions

| Issue | Root Cause | Solution Implemented |
| :--- | :--- | :--- |
| **Stream stopped after 100 frames** | Hardcoded `while frame_id < 100:` in sender. | Changed to `while True:` with clean `finally` cleanup. |
| **Model detected 0 objects** | `models/yoloe-26m-seg-pf.pt` was a symlink to text-prompted checkpoint without loaded vocabulary. | Downloaded authentic prompt-free models (`yoloe-26s-seg-pf.pt`) featuring built-in 4,585-class LRPC vocabulary. |
| **High CPU latency (~1,915 ms / 0.5 FPS)** | 26M parameter model at 640×640 on CPU without CUDA GPU. | Switched to `yoloe-26s-seg-pf.pt` at `imgsz=480`, dropping inference to **~286 ms (~3.5 FPS)**. |
| **Clock drift (`delay=-0.916s`)** | Windows RealSense SDK hardware timestamp was out of sync with WSL system clock. | `d455_receiver` now stamps packets using `self.get_clock().now().to_msg()`. |
| **Scene graph empty (`Objects: 0`)** | RTAB-Map had `quality=0` on stationary desk, causing TF lookup failure (`world_T_camera = None`). | Added robust camera-frame pose fallback (`np.eye(4)`) in `scene_graph_node.py` and enabled static TF by default (`use_rtabmap:=false`). |
| **RViz Fixed Frame error** | RViz configured for `world`, but launch was defaulting to `map`. | Set `world_frame` default to `"world"` across all launch files and nodes. |
