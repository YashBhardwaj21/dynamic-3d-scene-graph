# Stage 1: Intel RealSense D455 Hardware Configuration

## 1. Operating Point Rationale

The following operating points are **protected and experimentally tuned**:

| Parameter | Value | Classification | Rationale |
| :--- | :--- | :--- | :--- |
| **Color Stream** | 640×480 @ 30 FPS BGR8 | EXPERIMENTALLY-VALIDATED | Balances pixel density with YOLO/segmentation latency and USB 3 transfer bandwidth. |
| **Depth Stream** | 640×480 @ 30 FPS Z16 | EXPERIMENTALLY-VALIDATED | Matches color dimensions for 1:1 hardware spatial alignment (`rs.align`). |
| **IMU Accel** | 200 Hz | HARDWARE-DERIVED | Native BMI085 accelerometer sampling rate. |
| **IMU Gyro** | 200 Hz | HARDWARE-DERIVED | Native BMI085 gyroscope sampling rate. |
| **Depth Scale** | $\approx 0.001\text{ m/unit}$ | HARDWARE-DERIVED | Queried at runtime via `depth_sensor.get_depth_scale()`. |
| **Live Sync Tolerance** | 0.05 s (50 ms) | EXPERIMENTALLY-VALIDATED | D455 active stereo and rolling shutter capture exhibits up to 33 ms skew under load. |
| **TUM Sync Tolerance** | 0.02 s (20 ms) | DATASET CONSTANT | TUM benchmark dataset frames are paired within 20 ms. |
| **SLAM Pose Max Age** | 0.08 s (80 ms) | SAFETY LIMIT | Live freshness limit (<80ms accepted, >80ms rejected to prevent stale tracking backlog). |
| **Future TF Tolerance** | $1\times 10^{-4}\text{ s}$ | SAFETY LIMIT | Strictly rejects non-causal TF lookups that look ahead into the future. |

## 2. Dynamic Hardware Queries vs Hardcoded Constants

- **Intrinsics:** RealSense factory calibration is queried at startup from `color_profile.as_video_stream_profile().get_intrinsics()`.
- **Depth Scale:** `depth_sensor.get_depth_scale()` is queried on pipeline start and transmitted in packet metadata.
- **Hardware Metadata:** Queried using the RealSense `supports_frame_metadata()` pattern to prevent SDK crashes on unsupported firmware fields.
