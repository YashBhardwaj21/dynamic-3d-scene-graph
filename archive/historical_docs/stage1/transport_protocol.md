# Stage 1: Windows -> WSL2 Transport Protocol

## 1. Physical & Network Topology

- **Sender:** Windows 11 host running `tools/d455_sender.py`.
- **Receiver:** Ubuntu 22.04 on WSL2 running `ros2_ws/src/d455_bridge/d455_bridge/d455_receiver.py`.
- **Link:** Hyper-V Internal Virtual Switch (`vEthernet (WSL)`).
- **Socket Options:**
  - `TCP_NODELAY = 1`: Disables Nagle's algorithm to eliminate 40 ms packet batching latency.
  - `SO_SNDBUF = 4 MB` and `SO_RCVBUF = 4 MB`: Large kernel buffers prevent packet drops during instantaneous bursts.

## 2. Packet Wire Format (Protocol v2)

The wire framing is 100% backward-compatible:
```
+---------------------------+-----------------------------------+--------------------+--------------------+
| Header Length (4 bytes)   | JSON Metadata Header (UTF-8)      | RGB Bytes (BGR8)   | Depth Bytes (Z16)  |
| uint32 (Big-Endian)       | Length = Header Length            | Length = rgb_size  | Length = depth_size|
+---------------------------+-----------------------------------+--------------------+--------------------+
```

### Protocol v2 Header Schema
```json
{
  "protocol_version": 2,
  "type": "d455_rgbd",
  "session_id": "d455_b7a8c9_1728000000",
  "frame_id": 1420,
  "host_monotonic_time": 1234.5678,
  "host_wall_time": 1728000000.123,
  "color": {
    "timestamp": 123.456,
    "timestamp_domain": "hardware_clock",
    "width": 640,
    "height": 480,
    "format": "bgr8",
    "channels": 3,
    "bytes_per_channel": 1,
    "payload_size": 921600,
    "fx": 385.1, "fy": 385.1, "ppx": 320.0, "ppy": 240.0,
    "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    "distortion_model": "plumb_bob"
  },
  "depth": {
    "timestamp": 123.458,
    "timestamp_domain": "hardware_clock",
    "width": 640,
    "height": 480,
    "format": "z16",
    "channels": 1,
    "bytes_per_channel": 2,
    "payload_size": 614400,
    "depth_scale": 0.0010000000474974513,
    "is_aligned_to_color": true
  },
  "imu": [
    {
      "timestamp": 123.457,
      "domain": "hardware_clock",
      "accel": [0.05, 9.81, -0.12],
      "gyro": [0.002, -0.001, 0.003]
    }
  ],
  "rgb_depth_dt_ms": 2.0
}
```

## 3. Security Limits and Memory Allocation Safety

Before any memory allocation, the receiver validates:
1. `1 <= header_size <= 1 MB`: Rejects corrupted or malicious header size indicators.
2. `100 <= payload_size <= 10 MB`: Rejects impossible or oversized payload allocations.
3. `width * height * channels * bytes_per_channel == payload_size`: Strict consistency check.
4. Truncated socket reads immediately abort packet construction and initiate safe reconnection.
